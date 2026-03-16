"""Stage 2: Cross-Trajectory State Canonicalization (two-step).

Step 2a — State Definition:   single LLM call across all trajectories,
           defines the canonical state vocabulary.
Step 2b — Per-Trajectory Mapping:  one LLM call per trajectory,
           classifies each milestone into a canonical state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import STATE_DEFINER_SYSTEM, STATE_MAPPER_SYSTEM
from .trajectory_analyzer import TrajectoryAnalysis
from .utils import call_llm, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CanonicalState:
    state_id: str         # "S1", "S2", ...
    description: str
    type: str             # normal | terminal
    verification: list[str] = field(default_factory=list)


@dataclass
class StateMapping:
    traj_index: int
    milestone_id: int
    state_id: str
    reason: str = ""


@dataclass
class StateCanonicalizerResult:
    canonical_states: list[CanonicalState] = field(default_factory=list)
    mappings: list[StateMapping] = field(default_factory=list)
    removed_states: list[CanonicalState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
                {
                    "traj_index": m.traj_index,
                    "milestone_id": m.milestone_id,
                    "state_id": m.state_id,
                    "reason": m.reason,
                }
                for m in self.mappings
            ],
        }
        if self.removed_states:
            d["removed_states"] = [
                {
                    "state_id": s.state_id,
                    "description": s.description,
                    "type": s.type,
                    "reason": "defined by state generator but no milestone mapped to it",
                }
                for s in self.removed_states
            ]
        return d

    def trace_for(self, traj_index: int) -> list[str]:
        """Return the canonical state sequence for one trajectory."""
        relevant = [m for m in self.mappings if m.traj_index == traj_index]
        relevant.sort(key=lambda m: m.milestone_id)
        return [m.state_id for m in relevant]


# ---------------------------------------------------------------------------
# Canonicalizer
# ---------------------------------------------------------------------------

class StateCanonicalization:
    """Stage 2: two-step state canonicalization across trajectories."""

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
        """Run two-step canonicalization. Returns canonical states + mappings."""
        logger.info("state canonicalization started: %d trajectories", len(analyses))
        instance_id = analyses[0].instance_id

        # Step 2a: define canonical states (with retry on parse errors + contract check)
        has_resolved = any(a.resolved for a in analyses)
        last_2a_err: Exception | None = None
        for attempt_2a in range(1, 3):
            try:
                states = self._define_states(task_description, analyses, instance_id)
            except (ValueError, RuntimeError) as err:
                last_2a_err = err
                logger.warning("step 2a attempt %d failed: %s — retrying", attempt_2a, err)
                continue
            # Contract check: resolved trajectories need a terminal state
            has_terminal = any(s.type == "terminal" for s in states)
            if has_resolved and not has_terminal:
                last_2a_err = ValueError("no terminal state defined despite resolved trajectories")
                logger.warning("step 2a attempt %d: %s — retrying", attempt_2a, last_2a_err)
                continue
            break
        else:
            raise RuntimeError(f"step 2a failed after retries: {last_2a_err}")
        logger.info(
            "step 2a done: %d canonical states defined", len(states),
        )

        # Step 2b: per-trajectory mapping
        all_mappings: list[StateMapping] = []
        for i, a in enumerate(analyses):
            traj_mappings = self._map_trajectory(states, a, traj_index=i, instance_id=instance_id)
            all_mappings.extend(traj_mappings)
            logger.info(
                "step 2b: trajectory %d mapped (%d milestones)", i, len(traj_mappings),
            )

        result = StateCanonicalizerResult(canonical_states=states, mappings=all_mappings)

        # Validate
        self._validate(result)

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
    # Step 2a: State Definition
    # ------------------------------------------------------------------

    def _define_states(
        self,
        task_description: str,
        analyses: list[TrajectoryAnalysis],
        instance_id: str,
    ) -> list[CanonicalState]:
        """Single LLM call to define canonical state vocabulary."""
        prompt = self._build_definition_prompt(task_description, analyses)
        data = self._call_with_retry(
            prompt, system=STATE_DEFINER_SYSTEM, instance_id=instance_id, label="state_definition",
        )
        return self._parse_states(data)

    def _build_definition_prompt(
        self,
        task_description: str,
        analyses: list[TrajectoryAnalysis],
    ) -> str:
        """Build the user prompt for state definition."""
        sections: list[str] = []
        sections.append(f"## Task\n\n{task_description.strip()}\n")

        for i, a in enumerate(analyses):
            # Only show success milestones to state definer — error milestones
            # don't represent new capabilities and are handled in Step 2b.
            success_milestones = [m for m in a.milestones if m.outcome == "success"]
            status = "resolved=true" if a.resolved else "resolved=false"
            header = f"## Trajectory {i} ({status}, {len(success_milestones)} milestones)"
            lines = [header, ""]
            for m in success_milestones:
                parts = [
                    f"  M{m.milestone_id}:",
                    f"action=\"{m.action}\"",
                    f"| intent=\"{m.intent}\"",
                    f"| state=\"{m.state_reached}\"",
                ]
                if m.error_info:
                    parts.append(
                        f"| note=\"{m.error_info.symptom}\""
                    )
                lines.append(" ".join(parts))
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _parse_states(self, data: dict) -> list[CanonicalState]:
        """Parse and validate canonical states from LLM output."""
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
            if stype not in ("normal", "terminal"):
                raise ValueError(f"invalid state type '{stype}' for {sid}")
            if sid in seen_ids:
                raise ValueError(f"duplicate state_id: {sid}")
            seen_ids.add(sid)
            verification = [str(v).strip() for v in s.get("verification", []) if str(v).strip()]
            states.append(CanonicalState(
                state_id=sid, description=desc, type=stype, verification=verification,
            ))
        return states

    # ------------------------------------------------------------------
    # Step 2b: Per-Trajectory Mapping
    # ------------------------------------------------------------------

    def _map_trajectory(
        self,
        states: list[CanonicalState],
        analysis: TrajectoryAnalysis,
        *,
        traj_index: int,
        instance_id: str,
    ) -> list[StateMapping]:
        """Single LLM call to map one trajectory's milestones to canonical states.

        Retries on both JSON decode failures AND parse validation failures
        (unknown state_id, missing milestones).
        """
        prompt = self._build_mapping_prompt(states, analysis)
        label = f"mapping_traj{traj_index}"
        last_err: Exception | None = None
        for attempt in range(1, 3):
            resp = call_llm(
                prompt,
                system=STATE_MAPPER_SYSTEM,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            try:
                data = extract_json(resp.content)
                if not isinstance(data, dict):
                    raise ValueError("root must be a JSON object")
                return self._parse_mappings(
                    data, traj_index=traj_index, states=states, analysis=analysis,
                )
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err
                logger.warning(
                    "[%s] %s failed attempt %d/2: %s",
                    instance_id, label, attempt, err,
                )
        raise RuntimeError(
            f"[{instance_id}] {label} failed after retry: {last_err}"
        )

    def _build_mapping_prompt(
        self,
        states: list[CanonicalState],
        analysis: TrajectoryAnalysis,
    ) -> str:
        """Build the user prompt for per-trajectory mapping."""
        sections: list[str] = []

        # Canonical states
        sections.append("## Canonical States\n")
        for s in states:
            verification_str = "; ".join(s.verification) if s.verification else "N/A"
            sections.append(
                f"- **{s.state_id}** [{s.type}]: {s.description}\n"
                f"  Verification: {verification_str}"
            )

        # Trajectory milestones
        status = "resolved=true" if analysis.resolved else "resolved=false"
        sections.append(f"\n## Trajectory ({status}, {len(analysis.milestones)} milestones)\n")
        for m in analysis.milestones:
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
            sections.append(" ".join(parts))

        return "\n".join(sections)

    def _parse_mappings(
        self,
        data: dict,
        *,
        traj_index: int,
        states: list[CanonicalState],
        analysis: TrajectoryAnalysis,
    ) -> list[StateMapping]:
        """Parse and validate mappings from LLM output."""
        valid_ids = {s.state_id for s in states}

        raw_mappings = data.get("mappings")
        if not isinstance(raw_mappings, list):
            raise ValueError("mappings must be a list")

        mappings: list[StateMapping] = []
        seen_mids: set[int] = set()
        for m in raw_mappings:
            mid = int(m.get("milestone_id", -1))
            sid = str(m.get("state_id", "")).strip()
            reason = str(m.get("reason", "")).strip()

            if sid not in valid_ids:
                raise ValueError(
                    f"traj {traj_index}: mapping references unknown state_id: {sid}"
                )
            if mid in seen_mids:
                raise ValueError(
                    f"traj {traj_index}: duplicate mapping for milestone_id={mid}"
                )
            seen_mids.add(mid)
            mappings.append(StateMapping(
                traj_index=traj_index, milestone_id=mid, state_id=sid, reason=reason,
            ))

        # Coverage check for this trajectory
        expected_mids = {m.milestone_id for m in analysis.milestones}
        missing = expected_mids - seen_mids
        if missing:
            raise ValueError(
                f"traj {traj_index}: unmapped milestones: {sorted(missing)}"
            )

        return mappings

    # ------------------------------------------------------------------
    # Validation (post-mapping)
    # ------------------------------------------------------------------

    def _validate(self, result: StateCanonicalizerResult) -> None:
        """Post-mapping validation.

        Auto-removes unused canonical states and records them in
        ``result.removed_states`` for downstream traceability.
        (Coverage is already enforced per-trajectory in _parse_mappings.)
        """
        # Remove unused states
        used_ids = {m.state_id for m in result.mappings}
        kept: list[CanonicalState] = []
        removed: list[CanonicalState] = []
        for s in result.canonical_states:
            if s.state_id in used_ids:
                kept.append(s)
            else:
                removed.append(s)

        if removed:
            removed_ids = [s.state_id for s in removed]
            logger.info(
                "removed %d unused canonical state(s): %s", len(removed), removed_ids,
            )
            result.canonical_states = kept
            result.removed_states = removed

    # ------------------------------------------------------------------
    # LLM call + retry
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        prompt: str,
        *,
        system: str,
        instance_id: str,
        label: str,
    ) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, 3):
            resp = call_llm(
                prompt,
                system=system,
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
                    "[%s] %s JSON decode failed attempt %d/2: %s",
                    instance_id, label, attempt, err,
                )
        raise RuntimeError(
            f"[{instance_id}] {label} failed after retry: {last_err}"
        )
