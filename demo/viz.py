

from __future__ import annotations

import html

_GATEWAY_LABEL = {"exclusive": "XOR", "parallel": "AND", "inclusive": "OR"}
_START = "__start__"


def _esc(s: str) -> str:
    return html.escape(str(s)).replace("\n", " ")


def workflow_to_dot(
    workflow: dict,
    highlight_action_ids: set[str] | None = None,
    bad_action_ids: set[str] | None = None,
) -> str:
    """highlight_action_ids -> drawn green (e.g. the agent's correct trajectory).
    bad_action_ids -> drawn red (e.g. an off-path pick)."""
    actions = (workflow or {}).get("actions") or []
    gateways = (workflow or {}).get("gateways") or []
    hi = highlight_action_ids or set()
    bad = bad_action_ids or set()

    lines = [
        "digraph workflow {",
        '  rankdir=TB;',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
        f'  "{_START}" [label="start", shape=ellipse, style=filled, fillcolor="#d8f5d0"];',
    ]

    
    for a in actions:
        aid = a["id"]
        if aid in bad:
            fill, pen = "#f8d0d0", "#c0392b"
        elif aid in hi:
            fill, pen = "#d8f5d0", "#27ae60"
        else:
            fill, pen = "#eef2f7", "#34495e"
        label = _esc(a.get("name", aid))
        actor = a.get("actor")
        if actor:
            label += f"\\n({_esc(actor)})"
        lines.append(
            f'  "{aid}" [label="{label}", shape=box, style="rounded,filled", '
            f'fillcolor="{fill}", color="{pen}"];'
        )

    
    for g in gateways:
        gid = g["id"]
        gtype = _GATEWAY_LABEL.get(g.get("type", ""), "XOR")
        lines.append(
            f'  "{gid}" [label="{gtype}", shape=diamond, style=filled, '
            f'fillcolor="#fdf0d5", color="#b8860b", width=0.6, height=0.6];'
        )

    node_ids = {a["id"] for a in actions} | {g["id"] for g in gateways}

    
    for a in actions:
        if "start" in (a.get("predecessors") or []):
            lines.append(f'  "{_START}" -> "{a["id"]}";')

    
    for a in actions:
        for s in a.get("successors") or []:
            if s in node_ids:
                lines.append(f'  "{a["id"]}" -> "{s}";')

   
    end_n = 0
    for g in gateways:
        for b in g.get("branches") or []:
            nxt = b.get("next")
            cond = (b.get("condition") or "").strip()
            cond_attr = f' [label="{_esc(cond)}"]' if cond and cond.lower() != "unknown" else ""
            if nxt and nxt in node_ids:
                lines.append(f'  "{g["id"]}" -> "{nxt}"{cond_attr};')
            elif nxt is None:
                end_id = f"__end_{end_n}__"
                end_n += 1
                lines.append(f'  "{end_id}" [label="end", shape=ellipse, style=filled, fillcolor="#e8e8e8"];')
                lines.append(f'  "{g["id"]}" -> "{end_id}"{cond_attr};')

  
    fed: set[str] = set()
    for a in actions:
        fed.update(s for s in (a.get("successors") or []))
    for g in gateways:
        fed.update(b["next"] for b in (g.get("branches") or []) if b.get("next"))
    start_targets = [a["id"] for a in actions if "start" in (a.get("predecessors") or [])]
    for g in gateways:
        if "start" in (g.get("incoming_from") or []) and g["id"] not in fed:
            lines.append(f'  "{_START}" -> "{g["id"]}";')
            start_targets.append(g["id"])
    
    if not start_targets:
        for n in sorted(node_ids):
            if n not in fed:
                lines.append(f'  "{_START}" -> "{n}";')

    lines.append("}")
    return "\n".join(lines)
