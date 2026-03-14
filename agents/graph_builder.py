"""Stage 3: Algorithmic Graph Assembly.

Pure algorithm — zero LLM calls.

Takes the canonical states + mappings from Stage 2 and the original
trajectory analyses, and constructs a raw skill graph by counting
observed transitions in the canonical state space.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .state_canonicalizer import CanonicalState, StateCanonicalizerResult
from .trajectory_analyzer import TrajectoryAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for the raw graph
# ---------------------------------------------------------------------------

@dataclass
class RawNode:
    node_id: str          # same as canonical state_id (S1, E1, ...)
    raw_state: str        # description from canonicalization
    type: str             # start | intermediate | terminal | erroneous
    raw_verification: list[str] = field(default_factory=list)
    # Examples collected from contributing milestones
    action_examples: list[str] = field(default_factory=list)
    intent_examples: list[str] = field(default_factory=list)
    observation_examples: list[str] = field(default_factory=list)


@dataclass
class RawEdge:
    edge_id: int
    from_node: str        # node_id (state_id)
    to_node: str
    type: str             # normal | erroneous | rollback
    # Examples collected from contributing transitions
    action_examples: list[str] = field(default_factory=list)
    intent_examples: list[str] = field(default_factory=list)
    error_examples: list[str] = field(default_factory=list)


@dataclass
class TransitionContributor:
    traj_index: int
    from_milestone: int | None   # None for __START__ entries
    to_milestone: int


@dataclass
class TransitionMapping:
    edge_id: int
    from_node: str
    to_node: str
    type: str
    contributors: list[TransitionContributor] = field(default_factory=list)


@dataclass
class RawSkillGraph:
    instance_id: str
    nodes: list[RawNode] = field(default_factory=list)
    edges: list[RawEdge] = field(default_factory=list)
    transition_mappings: list[TransitionMapping] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "raw_state": n.raw_state,
                    "type": n.type,
                    "raw_verification": n.raw_verification,
                    "action_examples": n.action_examples,
                    "intent_examples": n.intent_examples,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "type": e.type,
                    "action_examples": e.action_examples,
                    "intent_examples": e.intent_examples,
                    "error_examples": e.error_examples,
                }
                for e in self.edges
            ],
        }

    def mappings_to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "edge_id": tm.edge_id,
                "transition": f"{tm.from_node} -> {tm.to_node}",
                "type": tm.type,
                "contributors": [
                    {
                        "traj_index": c.traj_index,
                        "from_milestone": c.from_milestone,
                        "to_milestone": c.to_milestone,
                    }
                    for c in tm.contributors
                ],
            }
            for tm in self.transition_mappings
        ]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Stage 3: build a raw skill graph from canonical states + transitions."""

    def build(
        self,
        instance_id: str,
        canon: StateCanonicalizerResult,
        analyses: list[TrajectoryAnalysis],
    ) -> RawSkillGraph:
        logger.info("graph assembly started")

        # --- Build milestone lookup: (traj_index, milestone_id) -> Milestone ---
        milestone_lookup: dict[tuple[int, int], Any] = {}
        for i, a in enumerate(analyses):
            for m in a.milestones:
                milestone_lookup[(i, m.milestone_id)] = m

        # --- Build nodes (type comes directly from Stage 2) ---
        nodes: dict[str, RawNode] = {}
        for s in canon.canonical_states:
            ntype = {"normal": "intermediate", "terminal": "terminal",
                     "erroneous": "erroneous"}.get(s.type, "intermediate")
            nodes[s.state_id] = RawNode(
                node_id=s.state_id,
                raw_state=s.description,
                type=ntype,
                raw_verification=list(s.verification),
            )

        # --- Collect milestone examples into nodes ---
        for mapping in canon.mappings:
            ms = milestone_lookup.get((mapping.traj_index, mapping.milestone_id))
            if not ms:
                continue
            node = nodes.get(mapping.state_id)
            if not node:
                continue
            if ms.action:
                node.action_examples.append(ms.action)
            if ms.intent:
                node.intent_examples.append(ms.intent)
            for obs in (ms.key_observations or []):
                node.observation_examples.append(obs)

        # --- Add virtual START node ---
        nodes["__START__"] = RawNode(
            node_id="__START__",
            raw_state="Start",
            type="start",
        )

        # --- Extract transitions by iterating each trajectory's canonical sequence ---
        # transition_key = (from_state_id, to_state_id)
        transition_data: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for i, a in enumerate(analyses):
            trace = canon.trace_for(i)
            if not trace or not a.milestones:
                continue

            # Entry transition: START -> first canonical state
            first_ms = milestone_lookup.get((i, a.milestones[0].milestone_id))
            transition_data[("__START__", trace[0])].append({
                "traj_index": i,
                "milestone": first_ms,
                "from_milestone_id": None,
                "to_milestone_id": a.milestones[0].milestone_id,
            })

            # Subsequent transitions between consecutive states
            for j in range(len(trace) - 1):
                from_sid = trace[j]
                to_sid = trace[j + 1]
                if from_sid == to_sid:
                    continue  # self-loop: skip

                from_mid = a.milestones[j].milestone_id
                to_mid = a.milestones[j + 1].milestone_id if j + 1 < len(a.milestones) else None
                to_ms = milestone_lookup.get((i, to_mid)) if to_mid else None

                transition_data[(from_sid, to_sid)].append({
                    "traj_index": i,
                    "milestone": to_ms,
                    "from_milestone_id": from_mid,
                    "to_milestone_id": to_mid,
                })

        # --- Build edges + transition mappings ---
        edges: list[RawEdge] = []
        transition_mappings: list[TransitionMapping] = []
        edge_id = 1
        for (from_sid, to_sid), examples in sorted(transition_data.items()):
            from_node = nodes.get(from_sid)
            to_node = nodes.get(to_sid)
            if not from_node or not to_node:
                continue

            etype = self._edge_type(from_node.type, to_node.type)

            action_examples: list[str] = []
            intent_examples: list[str] = []
            error_examples: list[str] = []
            contributors: list[TransitionContributor] = []

            for ex in examples:
                contributors.append(TransitionContributor(
                    traj_index=ex["traj_index"],
                    from_milestone=ex["from_milestone_id"],
                    to_milestone=ex["to_milestone_id"],
                ))
                ms = ex.get("milestone")
                if not ms:
                    continue
                if ms.action:
                    action_examples.append(ms.action)
                if ms.intent:
                    intent_examples.append(ms.intent)
                if ms.error_info and ms.error_info.symptom:
                    error_examples.append(ms.error_info.symptom)

            edges.append(RawEdge(
                edge_id=edge_id,
                from_node=from_sid,
                to_node=to_sid,
                type=etype,
                action_examples=action_examples,
                intent_examples=intent_examples,
                error_examples=error_examples[:5],
            ))
            transition_mappings.append(TransitionMapping(
                edge_id=edge_id,
                from_node=from_sid,
                to_node=to_sid,
                type=etype,
                contributors=contributors,
            ))
            edge_id += 1

        # --- Trim node examples ---
        for n in nodes.values():
            n.observation_examples = n.observation_examples[:5]

        # --- Connectivity check ---
        node_list = list(nodes.values())
        self._check_connectivity(node_list, edges)

        graph = RawSkillGraph(
            instance_id=instance_id,
            nodes=node_list,
            edges=edges,
            transition_mappings=transition_mappings,
        )
        logger.info(
            "graph assembly done: %d nodes, %d edges",
            len(graph.nodes), len(graph.edges),
        )
        return graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_type(from_type: str, to_type: str) -> str:
        if to_type == "erroneous":
            return "erroneous"
        if from_type == "erroneous":
            return "rollback"
        return "normal"  # includes start -> normal/initial/terminal

    @staticmethod
    def _check_connectivity(nodes: list[RawNode], edges: list[RawEdge]) -> None:
        """Log warnings for unreachable or dead-end nodes."""
        all_ids = {n.node_id for n in nodes}
        has_incoming = {e.to_node for e in edges}
        has_outgoing = {e.from_node for e in edges}

        initial_ids = {n.node_id for n in nodes if n.type == "start"}
        terminal_ids = {n.node_id for n in nodes if n.type == "terminal"}

        no_incoming = all_ids - has_incoming - initial_ids
        no_outgoing = all_ids - has_outgoing - terminal_ids - {n.node_id for n in nodes if n.type == "erroneous"}

        if no_incoming:
            logger.warning("nodes with no incoming edges (not initial): %s", sorted(no_incoming))
        if no_outgoing:
            logger.warning("nodes with no outgoing edges (not terminal/erroneous): %s", sorted(no_outgoing))
