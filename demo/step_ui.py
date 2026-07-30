
from __future__ import annotations

import pandas as pd
import streamlit as st


def _chips(names: list[str], empty: str = "(none)") -> str:
    if not names:
        return f"_{empty}_"
    return " ".join(f"`{n}`" for n in names)


def render_step(step: dict, picked_is_gold_valid: bool) -> None:
    """Render one rollout step: candidates + PRM/LLM/blended scores, tool firing, pick."""
    scores = step.get("scores") or {}
    final = scores.get("final") or {}
    prm = scores.get("prm_dist") or {}
    llm = scores.get("llm_dist") or {}
    picked = step.get("picked")

    st.markdown(f"#### Step {step['step']}")
    st.markdown("**Completed so far:** " + _chips(step.get("completed_before") or []))

    conds = step.get("conditions_active") or []
    cond_flat = sorted({c for grp in conds for c in grp})
    if cond_flat:
        st.markdown("**Branch conditions in play:** " + _chips(cond_flat))

    # ---- candidate scoreboard (only the ensemble configs produce scores)
    if final:
        rows = []
        for cand in sorted(final, key=lambda c: -final[c]):
            rows.append({
                "candidate": cand,
                "PRM": round(prm.get(cand, 0.0), 3),
                "LLM": round(llm.get(cand, 0.0), 3),
                "Blended": round(final.get(cand, 0.0), 3),
                "picked": "✓" if cand == picked else "",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "PRM": st.column_config.ProgressColumn("PRM score", min_value=0.0, max_value=1.0, format="%.3f"),
                "LLM": st.column_config.ProgressColumn("LLM score", min_value=0.0, max_value=1.0, format="%.3f"),
                "Blended": st.column_config.ProgressColumn("Blended", min_value=0.0, max_value=1.0, format="%.3f"),
                "picked": st.column_config.TextColumn("picked", width="small"),
            },
        )
        a = scores.get("alpha")
        meta = []
        if a is not None:
            meta.append(f"α={a} (blended = α·PRM + (1−α)·LLM)")
        if scores.get("top_score") is not None:
            meta.append(f"top={scores['top_score']:.3f}")
        if scores.get("margin") is not None:
            meta.append(f"margin={scores['margin']:.3f}")
        if meta:
            st.caption(" · ".join(meta))
    else:
        st.caption("This config produces a free-form / single pick (no per-candidate scores).")

   
    if "tool_called" in step:
        tool = step.get("tool") or {}
        if step["tool_called"]:
            useful = tool.get("tool_useful")
            head = "🔧 **Graph tool FIRED**" + (" — narrowed candidates" if useful else " — returned nothing usable")
            st.warning(head)
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"gate: top<{tool.get('tool_threshold')} or margin<{tool.get('tool_margin')}")
                st.markdown("**Valid next (from graph):** " + _chips(tool.get("valid_next") or []))
                st.markdown("**Narrowed to:** " + _chips(tool.get("narrowed") or []))
            with cols[1]:
                nf = tool.get("narrowed_final") or {}
                if nf:
                    st.markdown("**Re-ranked within narrowed set:**")
                    st.dataframe(
                        pd.DataFrame([{"candidate": k, "score": round(v, 3)} for k, v in
                                      sorted(nf.items(), key=lambda kv: -kv[1])],
                                     ),
                        hide_index=True, use_container_width=True,
                        column_config={"score": st.column_config.ProgressColumn(
                            "score", min_value=0.0, max_value=1.0, format="%.3f")},
                    )
        else:
            st.info("Graph tool not triggered — agent was confident enough.")

   
    valid_opts = step.get("valid_options") or []
    if picked_is_gold_valid:
        st.success(f"**Picked:** `{picked}`  ✅ on-path (a valid next action)")
    else:
        st.error(f"**Picked:** `{picked}`  ❌ off-path")
        st.caption("Valid options were: " + _chips(valid_opts))


def render_trajectory(rollout: dict, branches: list[dict], has_gold: bool) -> None:
    """Step 5: the completed trajectory graded green/red against the reference paths."""
    traj = rollout.get("completed_trajectory") or []
    status = rollout.get("status")
    match_type = rollout.get("match_type")
    matched = rollout.get("matched_branch")

    ref_label = "gold" if has_gold else "extracted-graph"
    c1, c2, c3 = st.columns(3)
    c1.metric("Rollout status", status)
    c2.metric(f"Match vs {ref_label}", match_type)
    c3.metric("Matched branch", "—" if matched is None else f"#{matched}")

    # color each step of the trajectory against the matched reference path
    ref_path = None
    if matched is not None and 0 <= matched < len(branches):
        ref_path = branches[matched]["path"]

    st.markdown("**Agent trajectory:**")
    if not traj:
        st.write("_(empty — agent went off-path on the first step)_")
    else:
        parts = []
        for i, name in enumerate(traj):
            ok = ref_path is not None and i < len(ref_path) and ref_path[i] == name
            color = "#1e7d32" if ok else "#b71c1c"
            bg = "#d8f5d0" if ok else "#f8d0d0"
            parts.append(
                f'<span style="background:{bg};color:{color};padding:3px 8px;'
                f'border-radius:6px;margin:2px;display:inline-block;font-family:monospace">{name}</span>'
            )
        st.markdown(" → ".join(parts), unsafe_allow_html=True)

    with st.expander(f"Reference {ref_label} branches ({len(branches)})"):
        for b in branches:
            mark = "  ⬅ matched" if b["branch"] == matched else ""
            st.markdown(f"**Branch {b['branch']}{mark}:** " + " → ".join(f"`{n}`" for n in b["path"]))
