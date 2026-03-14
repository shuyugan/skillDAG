"""Stage 2: Cross-Trajectory State Canonicalization.

Single LLM call — takes all trajectory analyses for one task and produces:
  1. A canonical state vocabulary (the nodes of the future graph).
  2. A mapping from every (traj, milestone) to a canonical state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import STATE_CANONICALIZER_SYSTEM
from .trajectory_analyzer import TrajectoryAnalysis
from .utils import call_llm, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CanonicalState:
    state_id: str         # "S1", "S2", ..., "E1", "E2", ...
    description: str
    type: str             # normal | terminal | erroneous
    verification: list[str] = field(default_factory=list)


@dataclass
class StateMapping:
    traj_index: int
    milestone_id: int
    state_id: str


@dataclass
class StateCanonicalizerResult:
    canonical_states: list[CanonicalState] = field(default_factory=list)
    mappings: list[StateMapping] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_states": [
                {
                    "state_id": s.state_id,
                    "description": s.description,
                    "type": s.type,
                    "verification": s.verification,
                }
                for s in self.canonical_states
            ],
            "mappings": [
                {"traj_index": m.traj_index, "milestone_id": m.milestone_id, "state_id": m.state_id}
                for m in self.mappings
            ],
        }

    def trace_for(self, traj_index: int) -> list[str]:
        """Return the canonical state sequence for one trajectory."""
        relevant = [m for m in self.mappings if m.traj_index == traj_index]
        relevant.sort(key=lambda m: m.milestone_id)
        return [m.state_id for m in relevant]


# ---------------------------------------------------------------------------
# Canonicalizer
# ---------------------------------------------------------------------------

class StateCanonicalization:
    """Stage 2: global state canonicalization across trajectories."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def canonicalize(
        self,
        task_description: str,
        analyses: list[TrajectoryAnalysis],
        output_path: str | Path | None = None,
    ) -> StateCanonicalizerResult:
        """Run canonicalization. Returns canonical states + mappings."""
        logger.info("state canonicalization started: %d trajectories", len(analyses))

        prompt = self._build_prompt(task_description, analyses)
        data = self._call_with_retry(prompt, analyses[0].instance_id)
        result = self._parse(data, analyses)

        if output_path is not None:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "canonicalization.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info(
            "canonicalization done: %d canonical states, %d mappings",
            len(result.canonical_states),
            len(result.mappings),
        )
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        task_description: str,
        analyses: list[TrajectoryAnalysis],
    ) -> str:
        """Build the user prompt for the canonicalization LLM call."""
        sections: list[str] = []
        sections.append(f"## Task\n\n{task_description.strip()}\n")

        for i, a in enumerate(analyses):
            status = "resolved=true" if a.resolved else "resolved=false"
            header = f"## Trajectory {i} ({status}, {len(a.milestones)} milestones)"
            lines = [header, ""]
            for m in a.milestones:
                parts = [
                    f"  M{m.milestone_id}:",
                    f"action=\"{m.action}\"",
                    f"| intent=\"{m.intent}\"",
                    f"| state=\"{m.state_reached}\"",
                    f"| outcome={m.outcome}",
                ]
                if m.error_info:
                    parts.append(
                        f"| error_type={m.error_info.error_type}"
                        f", symptom=\"{m.error_info.symptom}\""
                    )
                lines.append(" ".join(parts))
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # LLM call + retry
    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str, instance_id: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, 3):
            resp = call_llm(
                prompt,
                system=STATE_CANONICALIZER_SYSTEM,
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
                    "[%s] canonicalization JSON decode failed attempt %d/2: %s",
                    instance_id, attempt, err,
                )
        raise RuntimeError(
            f"[{instance_id}] state canonicalization failed after retry: {last_err}"
        )

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------

    def _parse(
        self,
        data: dict,
        analyses: list[TrajectoryAnalysis],
    ) -> StateCanonicalizerResult:
        """Validate and parse the LLM output."""
        # --- canonical_states ---
        raw_states = data.get("canonical_states")
        if not isinstance(raw_states, list) or not raw_states:
            raise ValueError("canonical_states must be a non-empty list")

        states: list[CanonicalState] = []
        seen_ids: set[str] = set()
        for s in raw_states:
            sid = str(s.get("state_id", "")).strip()
            desc = str(s.get("description", "")).strip()
            stype = str(s.get("type", "normal")).strip()
            if not sid or not desc:
                raise ValueError(f"canonical state missing id or description: {s}")
            if stype not in ("normal", "terminal", "erroneous"):
                raise ValueError(f"invalid state type '{stype}' for {sid}")
            if sid in seen_ids:
                raise ValueError(f"duplicate state_id: {sid}")
            seen_ids.add(sid)
            verification = [str(v).strip() for v in s.get("verification", []) if str(v).strip()]
            states.append(CanonicalState(
                state_id=sid, description=desc, type=stype, verification=verification,
            ))

        # --- mappings ---
        raw_mappings = data.get("mappings")
        if not isinstance(raw_mappings, list):
            raise ValueError("mappings must be a list")

        mappings: list[StateMapping] = []
        covered: set[tuple[int, int]] = set()
        for m in raw_mappings:
            ti = int(m.get("traj_index", -1))
            mid = int(m.get("milestone_id", -1))
            sid = str(m.get("state_id", "")).strip()

            if sid not in seen_ids:
                raise ValueError(f"mapping references unknown state_id: {sid}")
            key = (ti, mid)
            if key in covered:
                raise ValueError(f"duplicate mapping for traj_index={ti}, milestone_id={mid}")
            covered.add(key)
            mappings.append(StateMapping(traj_index=ti, milestone_id=mid, state_id=sid))

        # --- coverage check ---
        expected: set[tuple[int, int]] = set()
        for i, a in enumerate(analyses):
            for m in a.milestones:
                expected.add((i, m.milestone_id))

        missing = expected - covered
        if missing:
            logger.warning("canonicalization missing mappings: %s", sorted(missing))

        # --- state reference check ---
        used_ids = {m.state_id for m in mappings}
        unused = seen_ids - used_ids
        if unused:
            logger.warning("canonical states defined but unused: %s", sorted(unused))

        return StateCanonicalizerResult(canonical_states=states, mappings=mappings)
