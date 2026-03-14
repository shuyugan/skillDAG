"""Trajectory Normalizer for SWE-bench ATIF-v1.2 trajectories.

Converts raw JSON trajectories from harbor/mini-swe-agent into a structured
normalized format (dataclasses + markdown) suitable for downstream skill
extraction.

Trajectory JSON layout (ATIF-v1.2, mini-swe-agent):
  step_id=1  source=system   -> system prompt (ignored)
  step_id=2  source=user     -> task description
  step_id=3  source=agent    -> first agent turn (has `reasoning_content`)
  step_id=4+ source=agent    -> subsequent agent turns (`message` only)

Each agent step contains:
  - message:            full text response (THOUGHT + ```bash ...```)
  - reasoning_content:  (step_id=3 only) short initial reasoning from the model
  - tool_calls:         list[{tool_call_id, function_name, arguments}]
  - observation:        {results: [{content: str}]}
  - metrics:            token / cost info
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool invocation extracted from a trajectory step."""
    tool_call_id: str
    function_name: str
    command: str  # the shell command (for bash_command type)


@dataclass
class NormalizedStep:
    """One normalized agent step (thought -> action -> observation)."""
    step: int                       # 1-indexed normalized step number
    original_step_id: int           # original step_id from the trajectory
    thought: str                    # agent's reasoning / thought
    actions: list[ToolCall]         # tool calls made in this step
    observation: str                # concatenated observation results


@dataclass
class NormalizedTrajectory:
    """A fully normalized trajectory for one attempt at a task."""
    instance_id: str                # SWE-bench instance id (e.g. django__django-11740)
    session_id: str
    agent_name: str
    model_name: str
    task_description: str           # the user message (step_id=2)
    resolved: bool = False          # whether the task was resolved (reward=1)
    steps: list[NormalizedStep] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_steps: int = 0

    def to_markdown(self) -> str:
        """Render the normalized trajectory as readable markdown."""
        lines: list[str] = []
        status = "SUCCESS" if self.resolved else "FAILURE"
        lines.append(f"# Trajectory: {self.instance_id}")
        lines.append(f"")
        lines.append(f"- **Result:** {status}")
        lines.append(f"- **Total steps:** {self.total_steps}")
        lines.append("")
        lines.append(self.task_description)
        lines.append("")
        lines.append("---")
        lines.append("")

        for s in self.steps:
            lines.append(f"## Step {s.step}")
            lines.append("")

            # Thought
            lines.append("### Thought")
            lines.append("")
            lines.append(s.thought)
            lines.append("")

            # Action
            lines.append("### Action")
            lines.append("")
            for tc in s.actions:
                lines.append(f"**{tc.function_name}**")
                lines.append("```bash")
                lines.append(tc.command)
                lines.append("```")
                lines.append("")

            # Observation
            lines.append("### Observation")
            lines.append("")
            lines.append("```")
            lines.append(s.observation)
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TrajectoryNormalizer:
    """Parse and normalize ATIF-v1.2 trajectory JSON files.

    Usage::

        normalizer = TrajectoryNormalizer()
        traj = normalizer.normalize("path/to/attempt_folder")
        md = traj.to_markdown()
        # or
        normalizer.normalize_to_file("path/to/attempt_folder", "output.md")

    The attempt folder is expected to contain:
      - agent/trajectory.json
      - verifier/reward.txt  (contains 0 or 1)
    """

    # step_id where the agent's first turn starts
    AGENT_START_STEP_ID = 3

    def normalize(self, attempt_dir: str | Path) -> NormalizedTrajectory:
        """Load a trajectory from an attempt folder and return a NormalizedTrajectory.

        Args:
            attempt_dir: Path to the attempt folder containing
                         agent/trajectory.json and verifier/reward.txt.
        """
        attempt_dir = Path(attempt_dir)
        trajectory_path = attempt_dir / "agent" / "trajectory.json"
        reward_path = attempt_dir / "verifier" / "reward.txt"

        with open(trajectory_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Read reward (resolved status)
        resolved = False
        if reward_path.exists():
            reward_text = reward_path.read_text(encoding="utf-8").strip()
            resolved = reward_text == "1"

        # Extract metadata
        session_id = raw.get("session_id", "unknown")
        agent_info = raw.get("agent", {})
        agent_name = agent_info.get("name", "unknown")
        model_name = agent_info.get("model_name", "unknown")

        # Infer instance_id from directory name  (e.g. django__django-11740__6VqGSjQ)
        instance_id = self._infer_instance_id(attempt_dir)

        steps_raw = raw.get("steps", [])

        # Extract task description from step_id=2 (source=user)
        task_description = self._extract_task(steps_raw)
        output = Path(attempt_dir) / "task.txt"
        output.write_text(task_description, encoding="utf-8")

        # Normalize agent steps (step_id >= 3)
        normalized_steps: list[NormalizedStep] = []
        total_cost = 0.0
        norm_idx = 0

        for step in steps_raw:
            if step.get("source") != "agent":
                continue

            step_id = step["step_id"]
            norm_idx += 1

            # --- Thought ---
            thought = self._extract_thought(step)

            # --- Actions (tool_calls) ---
            actions = self._extract_actions(step)

            # --- Observation ---
            observation = self._extract_observation(step)

            # --- Metrics ---
            metrics = step.get("metrics", {})
            total_cost += metrics.get("cost_usd", 0.0)

            normalized_steps.append(NormalizedStep(
                step=norm_idx,
                original_step_id=step_id,
                thought=thought,
                actions=actions,
                observation=observation,
            ))

        return NormalizedTrajectory(
            instance_id=instance_id,
            session_id=session_id,
            agent_name=agent_name,
            model_name=model_name,
            task_description=task_description,
            resolved=resolved,
            steps=normalized_steps,
            total_cost_usd=total_cost,
            total_steps=len(normalized_steps),
        )

    def normalize_to_file(
        self,
        attempt_dir: str | Path,
        output_path: str | Path,
    ) -> NormalizedTrajectory:
        """Normalize a trajectory and write the markdown to a file."""
        traj = self.normalize(attempt_dir)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(traj.to_markdown(), encoding="utf-8")
        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_instance_id(attempt_dir: Path) -> str:
        """Infer SWE-bench instance_id from the directory name.

        Expected directory name pattern:
          <repo>__<id>__<hash>  (e.g. django__django-11740__6VqGSjQ)

        We strip the trailing __<hash> suffix to get the canonical instance id.
        """
        dir_name = attempt_dir.name  # e.g. django__django-11740__6VqGSjQ
        parts = dir_name.rsplit("__", 1)
        if len(parts) == 2:
            return parts[0]  # e.g. django__django-11740
        return dir_name

    @staticmethod
    def _extract_task(steps: list[dict]) -> str:
        """Extract the task description from the user step (step_id=2)."""
        for step in steps:
            if step.get("step_id") == 2 and step.get("source") == "user":
                msg = step.get("message", "")
                # Strip the boilerplate workflow/rules suffix to keep only the
                # actual issue description.  The task sits between
                # "Please solve this issue:" and the first "## Recommended Workflow"
                # or "## Important Rules" heading.
                match = re.search(
                    r"Please solve this issue:\s*(.*?)(?=\n## Recommended Workflow|\n## Important Rules|\n## Formatting)",
                    msg,
                    re.DOTALL,
                )
                if match:
                    return match.group(1).strip()
                return msg.strip()
        return ""

    @staticmethod
    def _extract_thought(step: dict) -> str:
        """Extract the thought/reasoning from an agent step.

        For step_id=3 (first agent turn), `reasoning_content` holds the
        concise initial reasoning.  The `message` field for step_id=3
        typically contains a long multi-block dump that got rejected by
        the format checker, so we prefer `reasoning_content`.

        For step_id>=4, we parse the THOUGHT from the `message` text.
        """
        step_id = step.get("step_id", 0)

        # First agent step: prefer reasoning_content
        if step_id == 3 and step.get("reasoning_content"):
            return step["reasoning_content"].strip()

        # Subsequent steps: extract THOUGHT from message
        message = step.get("message", "")

        # Try to extract text before the code block
        # Common patterns: "THOUGHT: ...\n\n```bash" or just prose before ```bash
        match = re.match(r"(.*?)```bash", message, re.DOTALL)
        if match:
            thought_text = match.group(1).strip()
            # Remove "THOUGHT:" prefix if present
            thought_text = re.sub(r"^THOUGHT:\s*", "", thought_text, flags=re.IGNORECASE)
            if thought_text:
                return thought_text

        # Fallback: return the full message (minus code blocks)
        cleaned = re.sub(r"```.*?```", "", message, flags=re.DOTALL).strip()
        return cleaned if cleaned else message.strip()
        # return message.strip()

    @staticmethod
    def _extract_actions(step: dict) -> list[ToolCall]:
        """Extract tool calls from a step."""
        tool_calls_raw = step.get("tool_calls", [])
        actions = []
        for tc in tool_calls_raw:
            args = tc.get("arguments", {})
            command = args.get("command", "")
            actions.append(ToolCall(
                tool_call_id=tc.get("tool_call_id", ""),
                function_name=tc.get("function_name", ""),
                command=command,
            ))
        return actions

    @staticmethod
    def _extract_observation(step: dict) -> str:
        """Extract observation text from a step's observation results."""
        obs = step.get("observation", {})
        results = obs.get("results", [])
        parts = []
        for r in results:
            content = r.get("content", "")
            parts.append(content)
        return "\n".join(parts)

if __name__ == "__main__":
    normalizer = TrajectoryNormalizer()
    normalizer.normalize_to_file("/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-10x5-mini/django__django-11740__6VqGSjQ", "/space3/shuyu/project/skillDAG/workspace/agents/django-11740.md")
    print("Normalized trajectory written to ./")