"""Stage 1: Per-Trajectory Milestone Extraction.

Single LLM call per trajectory — extracts an ordered milestone sequence
from the normalized trajectory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import MILESTONE_EXTRACTOR_SYSTEM
from .trajectory_normalizer import NormalizedTrajectory, TrajectoryNormalizer
from .utils import call_llm, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ErrorInfo:
    error_type: str
    symptom: str


@dataclass
class Milestone:
    milestone_id: int
    action: str
    intent: str
    state_reached: str
    outcome: str  # success | error | partial
    error_info: ErrorInfo | None = None
    key_observations: list[str] = field(default_factory=list)


@dataclass
class TrajectoryAnalysis:
    instance_id: str
    resolved: bool
    summary: str
    milestones: list[Milestone] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        milestones = []
        for m in self.milestones:
            md: dict[str, Any] = {
                "milestone_id": m.milestone_id,
                "action": m.action,
                "intent": m.intent,
                "state_reached": m.state_reached,
                "outcome": m.outcome,
            }
            if m.error_info:
                md["error_info"] = {
                    "error_type": m.error_info.error_type,
                    "symptom": m.error_info.symptom,
                }
            else:
                md["error_info"] = None
            md["key_observations"] = m.key_observations
            milestones.append(md)
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "summary": self.summary,
            "milestones": milestones,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryAnalysis:
        milestones: list[Milestone] = []
        for m in data.get("milestones", []):
            err_raw = m.get("error_info")
            error_info = None
            if isinstance(err_raw, dict) and err_raw.get("error_type"):
                error_info = ErrorInfo(
                    error_type=str(err_raw.get("error_type", "other_error")),
                    symptom=str(err_raw.get("symptom", "")),
                )
            milestones.append(Milestone(
                milestone_id=int(m.get("milestone_id", 0)),
                action=str(m.get("action", "")),
                intent=str(m.get("intent", "")),
                state_reached=str(m.get("state_reached", "")),
                outcome=str(m.get("outcome", "success")),
                error_info=error_info,
                key_observations=list(m.get("key_observations") or []),
            ))
        return cls(
            instance_id=str(data.get("instance_id", "")),
            resolved=bool(data.get("resolved", False)),
            summary=str(data.get("summary", "")),
            milestones=milestones,
        )

    def to_markdown(self) -> str:
        status = "SUCCESS" if self.resolved else "FAILURE"
        lines = [
            f"# Analysis: {self.instance_id}",
            "",
            f"- **Result:** {status}",
            f"- **Milestones:** {len(self.milestones)}",
            "",
            f"**Summary:** {self.summary}",
            "",
            "---",
            "",
        ]
        for m in self.milestones:
            lines.append(f"## Milestone {m.milestone_id}")
            lines.append("")
            lines.append(f"**Action:** {m.action}")
            lines.append(f"**Intent:** {m.intent}")
            lines.append(f"**State reached:** {m.state_reached}")
            lines.append(f"**Outcome:** {m.outcome}")
            if m.error_info:
                lines.append(f"**Error type:** {m.error_info.error_type}")
                lines.append(f"**Symptom:** {m.error_info.symptom}")
            if m.key_observations:
                lines.append("**Key observations:**")
                for obs in m.key_observations:
                    lines.append(f"- {obs}")
            lines.extend(["", "---", ""])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TrajectoryAnalyzer:
    """Stage 1: extract milestones from a single normalized trajectory."""

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze(
        self,
        trajectory: NormalizedTrajectory,
        output_dir: str | Path | None = None,
    ) -> TrajectoryAnalysis:
        """Run milestone extraction on one trajectory."""
        logger.info("[%s] milestone extraction started", trajectory.instance_id)

        payload = {
            "instance_id": trajectory.instance_id,
            "resolved": trajectory.resolved,
            "task": trajectory.task_description,
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "actions": [a.command for a in s.actions],
                    "observation": s.observation[:2200],
                }
                for s in trajectory.steps
            ],
        }

        data = self._call_with_retry(
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            instance_id=trajectory.instance_id,
        )

        analysis = self._parse(data, trajectory.instance_id, trajectory.resolved)

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "analysis.json").write_text(
                json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out / "analysis.md").write_text(analysis.to_markdown(), encoding="utf-8")

        logger.info(
            "[%s] extraction done: %d milestones", trajectory.instance_id, len(analysis.milestones)
        )
        return analysis

    def analyze_from_dir(self, attempt_dirs: list[str | Path]) -> list[TrajectoryAnalysis]:
        """Normalize + analyze multiple attempt directories."""
        normalizer = TrajectoryNormalizer()
        results: list[TrajectoryAnalysis] = []
        for d in attempt_dirs:
            d = Path(d)
            traj = normalizer.normalize(d)
            (d / "normalized_trajectory.md").write_text(traj.to_markdown(), encoding="utf-8")
            analysis = self.analyze(traj, output_dir=d)
            results.append(analysis)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(self, *, prompt: str, instance_id: str) -> dict:
        """Call LLM, retry once on JSON decode failure."""
        last_err: Exception | None = None
        for attempt in range(1, 3):
            resp = call_llm(
                prompt,
                system=MILESTONE_EXTRACTOR_SYSTEM,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            try:
                data = extract_json(resp.content)
                if not isinstance(data, dict):
                    raise ValueError("root must be a JSON object")
                return data
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err
                logger.warning(
                    "[%s] JSON decode failed attempt %d/2: %s", instance_id, attempt, err
                )
        raise RuntimeError(
            f"[{instance_id}] milestone extraction failed after retry: {last_err}"
        )

    def _parse(self, data: dict, instance_id: str, resolved: bool) -> TrajectoryAnalysis:
        """Validate and parse LLM output into TrajectoryAnalysis."""
        summary = str(data.get("summary", "")).strip()
        raw_milestones = data.get("milestones")
        if not isinstance(raw_milestones, list) or not raw_milestones:
            raise ValueError("milestones must be a non-empty list")

        milestones: list[Milestone] = []
        for i, m in enumerate(raw_milestones):
            mid = int(m.get("milestone_id", i + 1))
            outcome = str(m.get("outcome", "success")).strip()
            if outcome not in ("success", "error"):
                raise ValueError(f"milestone {mid}: invalid outcome '{outcome}'")

            error_info = None
            ei = m.get("error_info")
            if isinstance(ei, dict) and ei.get("error_type"):
                error_info = ErrorInfo(
                    error_type=str(ei.get("error_type", "other_error")),
                    symptom=str(ei.get("symptom", "")),
                )

            milestones.append(Milestone(
                milestone_id=mid,
                action=str(m.get("action", "")).strip(),
                intent=str(m.get("intent", "")).strip(),
                state_reached=str(m.get("state_reached", "")).strip(),
                outcome=outcome,
                error_info=error_info,
                key_observations=list(m.get("key_observations") or []),
            ))

        return TrajectoryAnalysis(
            instance_id=instance_id,
            resolved=resolved,
            summary=summary,
            milestones=milestones,
        )
