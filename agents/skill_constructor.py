"""Stage 4: Skill Constructor — generalization + end-to-end orchestrator.

Orchestrates the full pipeline:
  Stage 1  TrajectoryAnalyzer   (per-traj, N LLM calls)
  Stage 2  StateCanonicalization (cross-traj, 1 LLM call)
  Stage 3  GraphBuilder          (pure algorithm)
  Stage 4  Language generalization (1 LLM call)

Also provides construct_from_files for resuming from saved analyses.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph_builder import GraphBuilder, RawEdge, RawNode, RawSkillGraph
from .prompts import GRAPH_GENERALIZER_SYSTEM
from .state_canonicalizer import StateCanonicalization
from .trajectory_analyzer import TrajectoryAnalysis, TrajectoryAnalyzer
from .trajectory_normalizer import TrajectoryNormalizer
from .utils import call_llm, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Final output data classes
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: str          # S1 / E1 style
    state: str
    type: str             # start | intermediate | terminal | erroneous
    verification: list[str] = field(default_factory=list)


@dataclass
class Edge:
    edge_id: int
    from_node: str
    to_node: str
    type: str             # normal | erroneous | rollback
    thought: str
    actions: str
    errors: list[str] = field(default_factory=list)


@dataclass
class SkillDAG:
    instance_id: str
    skill_name: str
    description: str
    trigger_conditions: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "skill_name": self.skill_name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "state": n.state,
                    "type": n.type,
                    "verification": n.verification,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "type": e.type,
                    "thought": e.thought,
                    "actions": e.actions,
                    "errors": e.errors,
                }
                for e in self.edges
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Skill: {self.skill_name}",
            "",
            f"**Source:** {self.instance_id}",
            "",
            f"**Description:** {self.description}",
            "",
        ]
        if self.trigger_conditions:
            lines.append("**Trigger conditions:**")
            for tc in self.trigger_conditions:
                lines.append(f"- {tc}")
            lines.append("")

        lines.extend([
            f"- **Nodes:** {len(self.nodes)}",
            f"- **Edges:** {len(self.edges)}",
            "", "---", "",
            "## Nodes", "",
        ])

        for n in self.nodes:
            lines.append(f"### {n.node_id} [{n.type}]")
            lines.append("")
            lines.append(n.state)
            if n.verification:
                lines.append("")
                lines.append("**Verification:**")
                for v in n.verification:
                    lines.append(f"- {v}")
            lines.extend(["", "---", ""])

        lines.extend(["## Edges", ""])
        for e in self.edges:
            lines.append(f"### Edge {e.edge_id} [{e.type}]: {e.from_node} -> {e.to_node}")
            lines.append("")
            lines.append(f"**Thought:** {e.thought}")
            lines.append(f"**Actions:** {e.actions}")
            if e.errors:
                lines.append("**Errors:**")
                for err in e.errors:
                    lines.append(f"- {err}")
            lines.extend(["", "---", ""])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class SkillConstructor:
    """End-to-end pipeline orchestrator + Stage 4 generalization."""

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct_from_scratch(
        self,
        attempt_dirs: list[str | Path],
        output_path: str | Path | None = None,
    ) -> SkillDAG:
        """Full pipeline: normalize → analyze → canonicalize → build → generalize."""
        logger.info("construct_from_scratch: %d attempt dirs", len(attempt_dirs))

        # Stage 1: per-trajectory milestone extraction
        analyzer = TrajectoryAnalyzer(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        analyses = analyzer.analyze_from_dir(attempt_dirs)

        return self._construct_from_analyses(analyses, attempt_dirs, output_path)

    def construct_from_files(
        self,
        attempt_dirs: list[str | Path],
        output_path: str | Path | None = None,
    ) -> SkillDAG:
        """Resume from existing analysis.json files."""
        logger.info("construct_from_files: %d attempt dirs", len(attempt_dirs))
        analyses = [self._load_analysis(Path(d)) for d in attempt_dirs]
        return self._construct_from_analyses(analyses, attempt_dirs, output_path)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _construct_from_analyses(
        self,
        analyses: list[TrajectoryAnalysis],
        attempt_dirs: list[str | Path],
        output_path: str | Path | None,
    ) -> SkillDAG:
        instance_id = analyses[0].instance_id
        task = self._read_task(attempt_dirs[0])

        # Stage 2: cross-trajectory state canonicalization
        canon = StateCanonicalization(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
        ).canonicalize(
            task_description=task,
            analyses=analyses,
            output_path=output_path,
        )

        # Stage 3: algorithmic graph assembly
        raw_graph = GraphBuilder().build(
            instance_id=instance_id,
            canon=canon,
            analyses=analyses,
        )

        # Save raw graph + transition mappings before generalization
        if output_path is not None:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "raw_graph.json").write_text(
                json.dumps(raw_graph.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out / "mapping.json").write_text(
                json.dumps(raw_graph.mappings_to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Stage 4: language generalization
        skill = self._generalize(raw_graph, task)

        if output_path is not None:
            self._save(skill, output_path)

        return skill

    # ------------------------------------------------------------------
    # Stage 4: Generalization
    # ------------------------------------------------------------------

    def _generalize(self, raw_graph: RawSkillGraph, task_description: str) -> SkillDAG:
        """Single LLM call to generalize the raw graph into a reusable skill."""
        logger.info("language generalization started")

        # Exclude the virtual __START__ node from LLM payload — it's structural only.
        payload = {
            "instance_id": raw_graph.instance_id,
            "task_description": task_description,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "raw_state": n.raw_state,
                    "type": n.type,
                    "raw_verification": n.raw_verification,
                    "action_examples": n.action_examples[:5],
                    "intent_examples": n.intent_examples[:5],
                }
                for n in raw_graph.nodes
                if n.type != "start"
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "type": e.type,
                    "action_examples": e.action_examples[:5],
                    "intent_examples": e.intent_examples[:5],
                    "error_examples": e.error_examples[:5],
                }
                for e in raw_graph.edges
            ],
        }

        resp = call_llm(
            json.dumps(payload, ensure_ascii=False, indent=2),
            system=GRAPH_GENERALIZER_SYSTEM,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        data = extract_json(resp.content)

        skill_name = str(data.get("skill_name", "Unnamed Skill")).strip()
        description = str(data.get("description", "")).strip()
        trigger_conditions = [str(t).strip() for t in data.get("trigger_conditions", []) if str(t).strip()]

        # Build generalized nodes
        gen_nodes: dict[str, dict] = {}
        for row in data.get("nodes", []):
            nid = str(row.get("node_id", "")).strip()
            if nid:
                gen_nodes[nid] = row

        nodes: list[Node] = []
        for rn in raw_graph.nodes:
            gn = gen_nodes.get(rn.node_id, {})
            state = str(gn.get("state", "")).strip() or rn.raw_state
            verification = [str(v).strip() for v in gn.get("verification", []) if str(v).strip()]
            if not verification:
                verification = list(rn.raw_verification)
            nodes.append(Node(
                node_id=rn.node_id, state=state, type=rn.type,
                verification=verification,
            ))

        # Build generalized edges
        gen_edges: dict[int, dict] = {}
        for row in data.get("edges", []):
            eid = int(row.get("edge_id", 0))
            if eid > 0:
                gen_edges[eid] = row

        edges: list[Edge] = []
        for re_ in raw_graph.edges:
            ge = gen_edges.get(re_.edge_id, {})
            thought = str(ge.get("thought", "")).strip() or "; ".join(re_.intent_examples[:2]) or "Advance workflow"
            actions = str(ge.get("actions", "")).strip() or "; ".join(re_.action_examples[:2]) or "Apply next step"
            errors = [str(e).strip() for e in ge.get("errors", []) if str(e).strip()]

            edges.append(Edge(
                edge_id=re_.edge_id,
                from_node=re_.from_node,
                to_node=re_.to_node,
                type=re_.type,
                thought=thought,
                actions=actions,
                errors=errors,
            ))

        logger.info("generalization done: skill_name=%s", skill_name)
        return SkillDAG(
            instance_id=raw_graph.instance_id,
            skill_name=skill_name,
            description=description,
            trigger_conditions=trigger_conditions,
            nodes=nodes,
            edges=edges,
        )

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save(skill: SkillDAG, output_path: str | Path) -> None:
        folder = Path(output_path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dag.json").write_text(
            json.dumps(skill.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        meta_lines = [
            f"# {skill.skill_name}",
            "",
            skill.description,
            "",
        ]
        if skill.trigger_conditions:
            meta_lines.append("## Trigger Conditions")
            meta_lines.append("")
            for tc in skill.trigger_conditions:
                meta_lines.append(f"- {tc}")
            meta_lines.append("")
        (folder / "meta.md").write_text("\n".join(meta_lines), encoding="utf-8")
        logger.info("skill saved to %s", folder)

    @staticmethod
    def _read_task(attempt_dir: str | Path) -> str:
        task_path = Path(attempt_dir) / "task.txt"
        if task_path.exists():
            return task_path.read_text(encoding="utf-8")
        traj = TrajectoryNormalizer().normalize(attempt_dir)
        return traj.task_description

    @staticmethod
    def _load_analysis(attempt_dir: Path) -> TrajectoryAnalysis:
        json_path = attempt_dir / "analysis.json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return TrajectoryAnalysis.from_dict(data)
        raise FileNotFoundError(f"analysis.json not found in {attempt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_attempt_dirs = [
        "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini/django__django-11740__6VqGSjQ",
        "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini/django__django-11740__dn2GBmx",
        "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini/django__django-11740__f7FKnxW",
        "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini/django__django-11740__PvmC2sc",
        "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini/django__django-11740__x5bmc32",
    ]

    parser = argparse.ArgumentParser(description="Construct skill DAG (workspace_state pipeline)")
    parser.add_argument("attempt_dirs", nargs="*", default=None)
    parser.add_argument("-o", "--output-path", default="/space3/shuyu/project/skillDAG/workspace_state/skill_lib/test_1")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Run full pipeline including trajectory analysis (default: read existing analysis.json)")
    args = parser.parse_args()

    if not args.attempt_dirs:
        args.attempt_dirs = default_attempt_dirs

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    constructor = SkillConstructor(model=args.model)
    if args.from_scratch:
        skill = constructor.construct_from_scratch(args.attempt_dirs, output_path=args.output_path)
    else:
        skill = constructor.construct_from_files(args.attempt_dirs, output_path=args.output_path)

    print(skill.to_markdown())
