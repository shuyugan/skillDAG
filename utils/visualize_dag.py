"""Visualize a skill DAG from a dag.json file.

Adapted for workspace_state pipeline where node_ids are strings (S1, E1, ...).
The dag.json already contains explicit __START__ edges — no implicit edges needed.

Usage:
    python -m workspace_state.utils.visualize_dag path/to/dag.json [output.png]
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

NODE_STYLES = {
    "start":        {"fc": "#ecf0f1", "ec": "#7f8c8d", "ls": "solid"},
    "intermediate": {"fc": "#d6eaf8", "ec": "#2980b9", "ls": "solid"},
    "terminal":     {"fc": "#fef9e7", "ec": "#f39c12", "ls": "solid"},
    "erroneous":    {"fc": "#fadbd8", "ec": "#e74c3c", "ls": "dashed"},
}

EDGE_STYLES = {
    "normal":    {"color": "#2c3e50", "ls": "solid",  "lw": 1.6},
    "erroneous": {"color": "#e74c3c", "ls": "dashed", "lw": 1.6},
    "rollback":  {"color": "#2980b9", "ls": "dotted", "lw": 1.6},
}

# Layout tuning
WRAP_WIDTH = 24          # chars per line inside node boxes
X_STEP = 10.0            # horizontal gap between layers
Y_STEP = 6.0             # vertical gap between nodes in same layer
ERR_X_OFFSET = 0.6       # erroneous node horizontal offset (fraction of X_STEP)
ERR_Y_OFFSET = 1.4       # erroneous node vertical offset (fraction of Y_STEP)
NODE_FONT = 8.5
EDGE_LABEL_FONT = 8.0
TITLE_FONT = 12.0
DETAIL_FONT = 7.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = WRAP_WIDTH) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "-"
    return "\n".join(textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False))


def _load_dag(dag_path: Path) -> dict[str, Any]:
    with dag_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "nodes" not in data or "edges" not in data:
        raise ValueError("dag.json must contain 'nodes' and 'edges'")
    return data


# ---------------------------------------------------------------------------
# Graph construction — NO implicit edges, dag.json is authoritative
# ---------------------------------------------------------------------------

def _build_graph(data: dict[str, Any]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for node in data.get("nodes", []):
        nid = str(node["node_id"])
        g.add_node(nid, node_id=nid, state=str(node.get("state", "")),
                   type=str(node.get("type", "intermediate")))

    for edge in data.get("edges", []):
        eid = int(edge["edge_id"])
        u = str(edge["from_node"])
        v = str(edge["to_node"])
        for n in (u, v):
            if n not in g:
                g.add_node(n, node_id=n, state=f"Node {n}", type="intermediate")
        g.add_edge(u, v, key=eid, edge_id=eid,
                   type=str(edge.get("type", "normal")),
                   thought=str(edge.get("thought", "")),
                   actions=str(edge.get("actions", "")),
                   errors=[str(x) for x in edge.get("errors", [])])
    return g


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _compute_positions(g: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    normal_nodes = [n for n in g.nodes if g.nodes[n].get("type") not in ("erroneous",)]
    err_nodes = [n for n in g.nodes if g.nodes[n].get("type") == "erroneous"]

    # Build a DAG of normal-typed edges for layering
    base = nx.DiGraph()
    base.add_nodes_from(normal_nodes)
    for u, v, _k, ed in g.edges(keys=True, data=True):
        if ed.get("type") in ("erroneous",):
            continue
        if u in base and v in base:
            base.add_edge(u, v)

    # Break cycles if any
    if not nx.is_directed_acyclic_graph(base):
        order = sorted(base.nodes(), key=str)
        idx = {n: i for i, n in enumerate(order)}
        acyclic = nx.DiGraph()
        acyclic.add_nodes_from(base.nodes())
        for u, v in base.edges():
            if idx[u] < idx[v]:
                acyclic.add_edge(u, v)
        base = acyclic

    try:
        topo = list(nx.topological_sort(base))
    except Exception:
        topo = sorted(base.nodes(), key=str)

    # Assign layers (longest-path layering)
    layer: dict[str, int] = {}
    for n in topo:
        preds = [layer[p] for p in base.predecessors(n) if p in layer]
        layer[n] = (max(preds) + 1) if preds else 0

    for n in normal_nodes:
        layer.setdefault(n, 0 if g.nodes[n].get("type") == "start" else 1)

    # Group by layer
    groups: dict[int, list[str]] = {}
    for n in normal_nodes:
        groups.setdefault(layer[n], []).append(n)

    pos: dict[str, tuple[float, float]] = {}
    for lx in sorted(groups):
        nodes = sorted(groups[lx], key=str)
        n_rows = len(nodes)
        for i, n in enumerate(nodes):
            y = -(i - (n_rows - 1) / 2) * Y_STEP
            x = lx * X_STEP
            pos[n] = (x, y)

    # Place erroneous nodes: below and slightly right of their source
    anchor_buckets: dict[str, list[str]] = {}
    for en in sorted(err_nodes, key=str):
        sources = [
            u for u, _v, _k, ed in g.in_edges(en, keys=True, data=True)
            if u in pos
        ]
        anchor = sources[0] if sources else "__START__"
        anchor_buckets.setdefault(anchor, []).append(en)

    for anchor, nodes in anchor_buckets.items():
        ax, ay = pos.get(anchor, (0, 0))
        for i, en in enumerate(nodes):
            pos[en] = (ax + X_STEP * ERR_X_OFFSET, ay - Y_STEP * (ERR_Y_OFFSET + i * 1.2))

    return pos


# ---------------------------------------------------------------------------
# Node box geometry
# ---------------------------------------------------------------------------

def _node_box_half_size(node_id: str, node_state: str) -> tuple[float, float]:
    if node_id == "__START__":
        return (0.7, 0.7)
    label = f"[{node_id}] {_wrap(node_state)}"
    lines = label.split("\n")
    max_chars = max((len(line) for line in lines), default=10)
    half_w = max(2.0, max_chars * 0.072 + 0.4)
    half_h = max(0.8, len(lines) * 0.22 + 0.35)
    return (half_w, half_h)


def _boundary_point(cx: float, cy: float, tx: float, ty: float,
                    hw: float, hh: float) -> tuple[float, float]:
    dx, dy = tx - cx, ty - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return (cx, cy)
    sx = float("inf") if abs(dx) < 1e-9 else hw / abs(dx)
    sy = float("inf") if abs(dy) < 1e-9 else hh / abs(dy)
    scale = min(sx, sy)
    return (cx + dx * scale, cy + dy * scale)


# ---------------------------------------------------------------------------
# Edge data
# ---------------------------------------------------------------------------

def _edge_records(g: nx.MultiDiGraph) -> list[dict[str, Any]]:
    edges = []
    for u, v, k, ed in g.edges(keys=True, data=True):
        edges.append({
            "u": u, "v": v, "k": k,
            "edge_id": int(ed.get("edge_id", k)),
            "type": str(ed.get("type", "normal")),
            "thought": str(ed.get("thought", "")),
            "actions": str(ed.get("actions", "")),
            "errors": [str(x) for x in ed.get("errors", [])],
        })
    edges.sort(key=lambda e: e["edge_id"])
    return edges


def _assign_edge_radii(edges: list[dict[str, Any]]) -> dict[int, float]:
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for e in edges:
        by_pair.setdefault((e["u"], e["v"]), []).append(e)

    radii: dict[int, float] = {}
    for (u, v), group in by_pair.items():
        group.sort(key=lambda x: x["edge_id"])
        k = len(group)
        reverse_exists = (v, u) in by_pair
        for i, e in enumerate(group):
            base = {"normal": 0.0, "erroneous": 0.22, "rollback": -0.22}.get(e["type"], 0.0)
            spread = 0.15
            offset = (i - (k - 1) / 2.0) * spread if k > 1 else 0.0
            if k == 1 and reverse_exists and abs(base) < 1e-9:
                offset = 0.18 if str(u) < str(v) else -0.18
            radii[e["edge_id"]] = base + offset
    return radii


def _label_point(x0: float, y0: float, x1: float, y1: float,
                 rad: float) -> tuple[float, float]:
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return (mx, my)
    nxv, nyv = -dy / dist, dx / dist
    offset = rad * dist * 0.5
    return (mx + nxv * offset, my + nyv * offset)


# ---------------------------------------------------------------------------
# Edge detail panel
# ---------------------------------------------------------------------------

def _format_edge_details(edges: list[dict[str, Any]], wrap_w: int = 70) -> str:
    lines: list[str] = []
    for e in edges:
        eid = e["edge_id"]
        lines.append(f"{'─' * 60}")
        lines.append(f"E{eid}  [{e['type']}]  {e['u']} → {e['v']}")
        lines.append(f"")
        thought = _wrap(e["thought"], wrap_w)
        actions = _wrap(e["actions"], wrap_w)
        for tl in thought.split("\n"):
            lines.append(f"  Thought: {tl}" if tl == thought.split("\n")[0] else f"          {tl}")
        lines.append("")
        for al in actions.split("\n"):
            lines.append(f"  Actions: {al}" if al == actions.split("\n")[0] else f"          {al}")
        if e["errors"]:
            lines.append("")
            lines.append(f"  Errors:")
            for err in e["errors"]:
                for el in _wrap(err, wrap_w - 4).split("\n"):
                    lines.append(f"    • {el}" if el == _wrap(err, wrap_w - 4).split("\n")[0] else f"      {el}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def visualize_dag(dag_path: str | Path, output_path: str | Path | None = None,
                  dpi: int = 170) -> None:
    dag_path = Path(dag_path)
    output_path = Path(output_path) if output_path else dag_path.with_suffix(".png")

    data = _load_dag(dag_path)
    g = _build_graph(data)
    pos = _compute_positions(g)
    edges = _edge_records(g)
    radii = _assign_edge_radii(edges)
    details_text = _format_edge_details(edges)

    # Compute node boxes (data-space sizes)
    node_boxes: dict[str, tuple[float, float]] = {}
    for nid in g.nodes:
        state = str(g.nodes[nid].get("state", ""))
        node_boxes[nid] = _node_box_half_size(nid, state)

    # Figure sizing
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = (max(xs) - min(xs)) if xs else 10.0
    span_y = (max(ys) - min(ys)) if ys else 6.0
    detail_lines = len(details_text.splitlines())

    graph_w = max(14.0, span_x * 0.6 + 8.0)
    panel_w = max(8.0, 10.0)
    fig_w = graph_w + panel_w
    fig_h = max(10.0, span_y * 0.6 + 6.0, detail_lines * 0.145 + 2.0)

    fig, (ax, ax_side) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [graph_w, panel_w]},
    )

    ax.axis("off")
    margin_x = max(4.0, max(hw for hw in [b[0] for b in node_boxes.values()]) + 1.5)
    margin_y = max(3.0, max(hh for hh in [b[1] for b in node_boxes.values()]) + 1.5)
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    # ── Draw edges ──
    for e in edges:
        u, v = e["u"], e["v"]
        eid = e["edge_id"]
        etype = e["type"]
        style = EDGE_STYLES.get(etype, EDGE_STYLES["normal"])
        rad = radii.get(eid, 0.0)

        x0, y0 = pos[u]
        x1, y1 = pos[v]
        hw0, hh0 = node_boxes[u]
        hw1, hh1 = node_boxes[v]
        sx, sy = _boundary_point(x0, y0, x1, y1, hw0, hh0)
        tx, ty = _boundary_point(x1, y1, x0, y0, hw1, hh1)

        arrow = FancyArrowPatch(
            (sx, sy), (tx, ty),
            arrowstyle="-|>", mutation_scale=14.0,
            linewidth=style["lw"], linestyle=style["ls"], color=style["color"],
            connectionstyle=f"arc3,rad={rad}", alpha=0.9, zorder=1,
        )
        ax.add_patch(arrow)

        lx, ly = _label_point(sx, sy, tx, ty, rad)
        ax.text(lx, ly, f"E{eid}", fontsize=EDGE_LABEL_FONT,
                ha="center", va="center", color=style["color"], zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                      "edgecolor": style["color"], "linewidth": 0.5, "alpha": 0.9})

    # ── Draw nodes ──
    for nid in g.nodes:
        x, y = pos[nid]
        ntype = str(g.nodes[nid].get("type", "intermediate"))
        style = NODE_STYLES.get(ntype, NODE_STYLES["intermediate"])
        hw, hh = node_boxes[nid]

        if nid == "__START__":
            circ = plt.Circle(
                (x, y), radius=0.7,
                facecolor=style["fc"], edgecolor=style["ec"],
                linewidth=2.2, zorder=3,
            )
            ax.add_patch(circ)
            ax.text(x, y, "START", fontsize=9, fontweight="bold",
                    ha="center", va="center", color="#4d5656", zorder=4)
            continue

        rect = mpatches.FancyBboxPatch(
            (x - hw, y - hh), hw * 2, hh * 2,
            boxstyle="round,pad=0.12",
            facecolor=style["fc"], edgecolor=style["ec"],
            linewidth=2.0, linestyle=style["ls"], zorder=3,
        )
        ax.add_patch(rect)

        state = str(g.nodes[nid].get("state", ""))
        label = f"[{nid}] {_wrap(state)}"
        ax.text(x, y, label, ha="center", va="center",
                fontsize=NODE_FONT, linespacing=1.15, zorder=4)

    # ── Title ──
    title = data.get("skill_name", data.get("instance_id", "Skill DAG"))
    desc = _wrap(str(data.get("description", "")), width=100)
    ax.set_title(f"Skill DAG: {title}\n{desc}",
                 fontsize=TITLE_FONT, fontweight="bold", pad=16)

    # ── Legend ──
    handles: list[Any] = []
    for ntype in ["intermediate", "terminal", "erroneous"]:
        s = NODE_STYLES[ntype]
        handles.append(mpatches.Patch(facecolor=s["fc"], edgecolor=s["ec"],
                                      linestyle=s["ls"], label=f"Node: {ntype}"))
    for etype in ["normal", "erroneous", "rollback"]:
        s = EDGE_STYLES[etype]
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["ls"],
                                     linewidth=2.0, label=f"Edge: {etype}"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95, ncol=2)

    # ── Side panel ──
    ax_side.axis("off")
    n_nodes = len([n for n in data.get("nodes", []) if n.get("type") != "start"])
    n_edges = len(data.get("edges", []))
    header = f"Nodes: {n_nodes}  │  Edges: {n_edges}\n{'═' * 60}\n"
    ax_side.set_title("Edge Details", fontsize=11, fontweight="bold", pad=10)
    ax_side.text(0.0, 1.0, header + details_text,
                 transform=ax_side.transAxes, ha="left", va="top",
                 fontsize=DETAIL_FONT, family="monospace", linespacing=1.2)

    fig.tight_layout(w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {output_path}  (nodes={n_nodes}, edges={n_edges})")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        visualize_dag(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None, dpi=400)
    else:
        dag_file = "/space3/shuyu/project/skillDAG/workspace_state/skill_lib/django__django-11740/dag.json"
        out_file = "/space3/shuyu/project/skillDAG/workspace_state/skill_lib/django__django-11740/dag.png"
        visualize_dag(dag_file, out_file, dpi=400)
