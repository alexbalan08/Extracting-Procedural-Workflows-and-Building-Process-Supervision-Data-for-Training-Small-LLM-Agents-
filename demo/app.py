#this is out streamlit demo, mostly code generated with Claude Code


from __future__ import annotations

import json
import os

import streamlit as st

import agent_runner as AR
import annotations as ANN
import extraction as X
import step_ui
from pdf_utils import extract_text
from viz import workflow_to_dot

st.set_page_config(page_title="Procedural Workflow Pipeline — Demo", layout="wide")


ss = st.session_state
ss.setdefault("procedure_text", "")
ss.setdefault("pdf_meta", None)
ss.setdefault("source_name", None)
ss.setdefault("file_index", None)      # set when the text is a held-out procedure -> gold available
ss.setdefault("extraction", None)      # full result dict from Step 2
ss.setdefault("workflow", None)        # the extracted JSON graph
ss.setdefault("rollout", None)         # Step 4 rollout (live or replayed)
ss.setdefault("branches", None)        # reference paths for grading
ss.setdefault("has_gold", False)       # whether grading is vs true gold
ss.setdefault("run_label", None)       # description of the config that produced rollout
ss.setdefault("step_ptr", 0)


def reset_downstream():
    """Any new text/extraction invalidates later steps."""
    ss.extraction = None
    ss.workflow = None
    reset_rollout()


def reset_rollout():
    ss.rollout = None
    ss.branches = None
    ss.run_label = None
    ss.step_ptr = 0



st.title("From Manuals to Reasoning Traces: create data for your own planning agents")
st.caption("End-to-end pipeline demo — PDF → JSON → workflow graph → agent deplyment → following and executing procedures")

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        "1. **Upload & extract text**\n"
        "2. **Extract workflow graph**\n"
        "3. Visualize graph\n"
        "4. Run planning agent\n"
        "5. Visualize trajectory\n"
        "6. Label & save dataset"
    )
    st.divider()
    st.subheader("OpenAI")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        st.success("OPENAI_API_KEY found in environment")
    api_key = st.text_input(
        "API key", value=env_key, type="password",
        help="Used by the live extractor (Step 2). Falls back to OPENAI_API_KEY env var.",
    )
    model = st.text_input("Extractor model", value="gpt-5.4-mini")
    st.divider()
    st.caption("M3/M4  needs a CUDA GPU. On the Mac, use replay from a saved run")

# =================================================================== STEP 1
st.header("Step 1 · Get the procedure text")

source = st.radio(
    "Source", ["Upload PDF", "Paste text", "Pick held-out procedure (has gold)"],
    horizontal=True, label_visibility="collapsed",
    help="Where the procedure text comes from. Upload/Paste = any new procedure (no gold "
         "to grade against). Held-out = one of the 50 test procedures, which has a "
         "human-annotated gold graph so Step 5 can grade the agent green/red.",
)

if source == "Upload PDF":
    uploaded = st.file_uploader("Upload a PDF containing a procedure", type=["pdf"])
    if uploaded is not None and st.button("Extract text", type="primary"):
        with st.spinner("Parsing PDF…"):
            text, meta = extract_text(uploaded.getvalue())
        ss.procedure_text, ss.pdf_meta, ss.source_name, ss.file_index = text, meta, uploaded.name, None
        reset_downstream()

elif source == "Paste text":
    txt = st.text_area("Procedure text", value=ss.procedure_text, height=220)
    if st.button("Use this text", type="primary") and txt.strip():
        ss.procedure_text = txt.strip()
        ss.pdf_meta = {"parser": "manual", "n_pages": 0, "errors": {}}
        ss.source_name, ss.file_index = "pasted text", None
        reset_downstream()

else:  # held-out picker — gives us gold for grading in Step 5
    idx = X.held_out_index()
    if not idx:
        st.error("No held_out.json found.")
    else:
        labels = {f"#{r['file_index']} · {r['n_actions']} actions · {r['preview']}": r["file_index"] for r in idx}
        choice = st.selectbox(
            "Held-out procedure", list(labels.keys()),
            help="50 test procedures (shortest first). Label shows file index, action "
                 "count, and a text preview. These come with gold graphs for grading.",
        )
        if st.button("Load procedure", type="primary"):
            fi = labels[choice]
            rec = X.held_out_record(fi)
            ss.procedure_text = rec["procedure_text"]
            ss.pdf_meta = {"parser": "held-out", "n_pages": 0, "errors": {}}
            ss.source_name, ss.file_index = f"held-out #{fi}", fi
            reset_downstream()

# ---- show text
if ss.procedure_text:
    meta = ss.pdf_meta or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", ss.source_name or "—")
    c2.metric("Parser", meta.get("parser") or "—")
    c3.metric("Characters", f"{len(ss.procedure_text):,}")
    c4.metric("Gold available", "yes ✓" if ss.file_index is not None else "no")
    if meta.get("errors"):
        st.warning("Parser fallbacks: " + "; ".join(f"{k}: {v}" for k, v in meta["errors"].items()))
    edited = st.text_area("Extracted procedure text (editable)", value=ss.procedure_text, height=200)
    if edited != ss.procedure_text:
        ss.procedure_text = edited
        reset_downstream()
else:
    st.info("Choose a source above to begin.")

# =================================================================== STEP 2
if ss.procedure_text:
    st.divider()
    st.header("Step 2 · Extract the workflow graph")
    st.caption("Runs the real pipeline: 3-shot prompt → structural checker → LLM semantic checker → self-refine loop.")

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        max_attempts = st.number_input(
            "Max self-refine attempts", 1, 5, 3,
            help="How many times the extractor may re-try. After each attempt the checkers "
                 "report issues, which are fed back into the prompt so the model fixes them. "
                 "More attempts = higher quality in general, but more API calls / cost.",
        )
    with opt2:
        use_llm_checker = st.toggle(
            "LLM semantic checker", value=True,
            help="The 'critic': a second LLM pass that judges whether the extracted graph "
                 "actually matches the procedure's meaning (missed steps, wrong branch "
                 "conditions, etc.) and asks for a re-extraction if not. Uses Reflexion memory "
                 "to remember past mistakes across procedures many past runs.",
        )
    with opt3:
        use_struct = st.toggle(
            "Structural checker", value=True,
            help="Fast rule-based validation of the graph: no standalone actions, gateways "
                 "well-formed, edges point to real nodes, etc. Catches malformed JSON before "
                 "the more expensive LLM critic runs.",
        )

    go_live = st.button(
        "Run extraction (live process)", type="primary",
        help="Calls the real extraction pipeline live via OpenAI on the text above. "
             "Needs an API key in the sidebar.",
    )

    if go_live:
        key = api_key or env_key
        if not key:
            st.error("No OpenAI API key. Add it in the sidebar or set OPENAI_API_KEY.")
        else:
            try:
                with st.spinner(f"Extracting with {model}… (structural + semantic self-refine)"):
                    client = X.make_client(key)
                    result = X.run_extraction(
                        ss.procedure_text, client, model=model,
                        max_attempts=int(max_attempts),
                        use_llm_checker=use_llm_checker,
                        use_structural_checker=use_struct,
                        file_index=ss.file_index,
                    )
                ss.extraction = result
                ss.workflow = result.get("workflow")
            except Exception as e:  # noqa: BLE001
                st.error(f"Extraction failed: {type(e).__name__}: {e}")


# ---- show extraction result
if ss.extraction:
    r = ss.extraction
    wf = r.get("workflow") or {}
    n_actions = len(wf.get("actions") or [])
    n_gateways = len(wf.get("gateways") or [])
    n_states = len(r.get("execution_states") or [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actions", n_actions)
    m2.metric("Gateways", n_gateways)
    m3.metric("Attempts used", r.get("attempt", "—"))
    m4.metric("Output tokens", f"{r.get('completion_tokens', 0):,}")

    if r.get("remaining_issues"):
        st.warning("Unresolved checker issues after final attempt: "
                   + "; ".join(r["remaining_issues"]))

    tab_json, tab_reason, tab_states = st.tabs(["Workflow JSON", "Reasoning trace", "Execution states"])
    with tab_json:
        st.json(wf)
    with tab_reason:
        st.markdown(r.get("reasoning") or "_(no reasoning captured)_")
    with tab_states:
        st.caption(f"{n_states} deterministic execution states enumerated from the graph "
                   "(these drive the agent's valid-next-action set).")
        st.json(r.get("execution_states") or [])

    st.success("Graph extracted.")

# =================================================================== STEP 3
if ss.workflow:
    st.divider()
    st.header("Step 3 · Visualize the workflow graph")
    st.caption("Boxes = actions · diamonds = gateways (XOR/AND/OR) · green = start · "
               "edges follow the same construction the agent traverses.")

    # gold graph is only available for held-out procedures
    gold_wf = None
    if ss.file_index is not None:
        gold_rec = X.held_out_record(ss.file_index)
        gold_wf = (gold_rec or {}).get("workflow")

    def _render(wf, title):
        try:
            st.graphviz_chart(workflow_to_dot(wf), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render {title}: {type(e).__name__}: {e}")

    if gold_wf:
        compare = st.toggle(
            "Compare with gold graph", value=True,
            help="Show the human-annotated gold graph next to the extracted one. "
                 "Differences between them are exactly the extraction errors the agent "
                 "must cope with in 'predicted' mode.",
        )
        if compare:
            g_actions = len(gold_wf.get("actions") or [])
            g_gw = len(gold_wf.get("gateways") or [])
            p_actions = len((ss.workflow.get("actions") or []))
            p_gw = len((ss.workflow.get("gateways") or []))
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Extracted** · {p_actions} actions · {p_gw} gateways")
                _render(ss.workflow, "extracted graph")
            with right:
                st.markdown(f"**Gold (held-out)** · {g_actions} actions · {g_gw} gateways")
                _render(gold_wf, "gold graph")
            if (p_actions, p_gw) != (g_actions, g_gw):
                st.caption(f"⚠️ Node-count mismatch — extracted has "
                           f"{p_actions - g_actions:+d} actions, {p_gw - g_gw:+d} gateways vs gold.")
        else:
            _render(ss.workflow, "extracted graph")
    else:
        _render(ss.workflow, "extracted graph")

# ---- human-in-the-loop labeling of the extracted graph (data flywheel)
if ss.workflow:
    with st.expander("✍️ Label this extraction (human-in-the-loop)"):
        st.caption("Mark each extracted action correct or wrong, optionally giving the right "
                   "name. Saves accumulate into a JSONL dataset for retraining. Defaults to "
                   "'correct' — just flip the wrong ones.")
        _acts = ss.workflow.get("actions") or []
        _gws = ss.workflow.get("gateways") or []
        _kp = f"ext_{ss.source_name}_{ss.file_index}"
        with st.form(_kp):
            _action_labels = {}
            for _a in _acts:
                _c1, _c2, _c3 = st.columns([3, 2, 3])
                _c1.markdown(f"**{_a.get('name', _a['id'])}**")
                _lbl = _c2.radio("label", ["correct", "wrong"], horizontal=True,
                                 key=f"al_{_kp}_{_a['id']}", label_visibility="collapsed")
                _cor = _c3.text_input("corrected", value="", key=f"ac_{_kp}_{_a['id']}",
                                      label_visibility="collapsed",
                                      placeholder="corrected name (optional)")
                _action_labels[_a["id"]] = {"name": _a.get("name"), "label": _lbl,
                                            "corrected_name": _cor.strip() or None}
            _gw_labels = {}
            if _gws:
                st.markdown("**Gateways / branches**")
                for _g in _gws:
                    _g1, _g2 = st.columns([3, 2])
                    _g1.markdown(f"`{_g['id']}` ({_g.get('type')})")
                    _gw_labels[_g["id"]] = _g2.radio("label", ["correct", "wrong"], horizontal=True,
                                                     key=f"gl_{_kp}_{_g['id']}",
                                                     label_visibility="collapsed")
            _notes = st.text_area("Notes (optional)", key=f"an_{_kp}")
            _saved = st.form_submit_button("💾 Save extraction labels", type="primary")
        if _saved:
            ANN.append_record({
                "kind": "extraction_labels",
                "source": ss.source_name,
                "file_index": ss.file_index,
                "has_gold": ss.file_index is not None,
                "procedure_text": ss.procedure_text,
                "workflow": ss.workflow,
                "action_labels": _action_labels,
                "gateway_labels": _gw_labels,
                "notes": _notes.strip(),
            })
            _nw = sum(1 for _v in _action_labels.values() if _v["label"] == "wrong")
            st.success(f"Saved · {len(_action_labels)} actions labeled ({_nw} marked wrong). "
                       "Dataset updated below.")

# =================================================================== STEP 4
if ss.workflow:
    st.divider()
    st.header("Step 4 · Run the planning agent step by step")

    exec_mode = st.radio(
        "Execution", ["Live inference", "Replay saved run"], horizontal=True,
        help="Live = run the chosen agent now (M3/M4 need a CUDA GPU). "
             "Replay = step through a saved run from Agent_testing/results/ — same real "
             "scores, works on any machine.",
    )

    if exec_mode == "Live inference":
        cfg_label = st.selectbox(
            "Agent configuration", list(AR.CONFIGS.keys()),
            index=list(AR.CONFIGS.keys()).index("M4 · Agentic ensemble (+ graph tool)"),
            help="The four ablation methods, each adding one layer:\n"
                 "• M1 Llama bare — just the procedure text, free-form next action.\n"
                 "• M2 Llama + actions — given the extracted action list, must pick one (kills hallucinations).\n"
                 "• M3 Ensemble — blends the base Llama with the specialised PRM (the planning reward model).\n"
                 "• M4 Agentic ensemble — M3 plus a graph tool the agent consults when unsure.\n"
                 "OpenAI configs are frontier-model baselines for comparison.",
        )
        cfg = AR.CONFIGS[cfg_label]

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            grade_mode = st.selectbox(
                "Graph the agent sees", ["predicted", "gold"],
                help="Which graph the agent uses as input.\n"
                     "• predicted = the graph just extracted in Step 2 — real deployment quality.\n"
                     "• gold = the human-annotated graph — the agent's ceiling if extraction were perfect.\n"
                     "Grading in Step 5 is always against gold; the gap between the two is the cost of "
                     "extraction errors. (gold needs a held-out procedure.)",
            )
            if grade_mode == "gold" and ss.file_index is None:
                st.caption("⚠️ gold needs a held-out procedure; will use predicted.")
                grade_mode = "predicted"
        with p2:
            alpha = st.slider(
                "α (PRM weight)", 0.0, 1.0, 0.90, 0.05,
                disabled=cfg["key"] not in ("ensemble", "agentic"),
                help="Blend weight in  blended = α·PRM + (1−α)·LLM.\n"
                     "α=1.0 → trust only the specialised PRM; α=0.0 → only the base Llama. "
                     "Higher α leans on the trained reward model. (M3/M4 only.)",
            )
        with p3:
            tool_threshold = st.slider(
                "tool threshold", 0.0, 1.0, 0.45, 0.05,
                disabled=cfg["key"] != "agentic",
                help="M4 only. The graph tool fires when the top blended score is BELOW this "
                     "value — i.e. the agent isn't confident about any candidate. "
                     "Higher = the tool fires more often.",
            )
        with p4:
            tool_margin = st.slider(
                "tool margin", 0.0, 1.0, 0.20, 0.05,
                disabled=cfg["key"] != "agentic",
                help="M4 only. The graph tool also fires when the gap between the top two "
                     "candidates is BELOW this value — i.e. a close call between options "
                     "(typically at a gateway). Higher = the tool fires more often.",
            )
        openai_model_agent = st.text_input(
            "OpenAI model (for OpenAI configs)", value="gpt-5.4-mini",
            help="Model name for the OpenAI baseline agents.",
        ) if cfg["openai"] else "gpt-5.4-mini"

        if cfg["gpu"]:
            st.info("ℹ️ This config loads Llama-3.1-8B (4-bit) + PRM LoRA — runs on your GPU box, "
                    "not on the Mac. Record the video there, or use **Replay saved run** here.")

        if st.button("Run agent", type="primary"):
            try:
                with st.spinner(f"Running {cfg_label}…"):
                    held = X.held_out_record(ss.file_index) if ss.file_index is not None else None
                    case, has_gold = AR.build_case(ss.extraction, ss.procedure_text, ss.file_index, held)
                    agent = AR.make_agent(cfg["key"], alpha=alpha, tool_threshold=tool_threshold,
                                          tool_margin=tool_margin, openai_model=openai_model_agent)
                    rollout = AR.run_live(case, agent, mode=grade_mode,
                                          give_candidates=cfg["cands"])
                    ss.rollout = rollout
                    ss.branches = AR.gold_branches(case)
                    ss.has_gold = has_gold
                    ss.run_label = f"{cfg_label} · {grade_mode} · {'live'}"
                    ss.step_ptr = 0
            except Exception as e:  # noqa: BLE001
                st.error(f"Agent run failed: {type(e).__name__}: {e}")

    else:  # Replay saved run
        files = AR.list_result_files()
        if not files:
            st.error("No saved runs in Agent_testing/results/.")
        elif ss.file_index is None:
            st.warning("Replay needs a held-out procedure (so the saved record matches). "
                       "Pick one in Step 1.")
        else:
            # default to an agentic-ensemble predicted file if present
            default_i = next((i for i, f in enumerate(files)
                              if "agentic_ensemble_predicted" in f), 0)
            rfile = st.selectbox(
                "Saved run", files, index=default_i,
                help="A pre-computed inference run. Filename encodes the method, mode "
                     "(predicted/gold) and params, e.g. agentic_ensemble_predicted_"
                     "alpha0.90_t0.45_m0.20. We load this procedure's record from it.",
            )
            if st.button("Load saved run", type="primary"):
                rec = AR.load_replay(rfile, ss.file_index)
                if rec is None:
                    st.error(f"This file has no record for held-out #{ss.file_index}.")
                else:
                    ss.rollout = rec["rollout"]
                    ss.branches = rec.get("branches") or []
                    ss.has_gold = "_gold" in rfile or "gold" in rec.get("eval_mode", "")
                    ss.run_label = f"{rfile} · replay"
                    ss.step_ptr = 0

# ---- step-by-step viewer
if ss.rollout:
    steps = ss.rollout.get("steps") or []
    st.caption(f"Showing: **{ss.run_label}**  ·  {len(steps)} steps")
    if not steps:
        st.warning("Rollout produced no steps (agent went off-path immediately or no candidates).")
    else:
        nav1, nav2, nav3, nav4 = st.columns([1, 1, 3, 2])
        with nav1:
            if st.button("◀ Prev", disabled=ss.step_ptr == 0):
                ss.step_ptr = max(0, ss.step_ptr - 1)
        with nav2:
            if st.button("Next ▶", disabled=ss.step_ptr >= len(steps) - 1):
                ss.step_ptr = min(len(steps) - 1, ss.step_ptr + 1)
        with nav3:
            ss.step_ptr = st.slider("Step", 1, len(steps), ss.step_ptr + 1) - 1
        with nav4:
            show_all = st.toggle(
                "Show all steps", value=False,
                help="Off = one step at a time (use Prev/Next/slider) — best for a live "
                     "walkthrough. On = render every step stacked, for a scrollable overview.",
            )

        if show_all:
            for s in steps:
                step_ui.render_step(s, s.get("is_valid", False))
                st.divider()
        else:
            step_ui.render_step(steps[ss.step_ptr], steps[ss.step_ptr].get("is_valid", False))

# =================================================================== STEP 5
if ss.rollout:
    st.divider()
    st.header("Step 5 · Final trajectory vs " + ("gold" if ss.has_gold else "extracted graph"))
    if not ss.has_gold:
        st.caption("No gold for this procedure — grading against the agent's own extracted graph. "
                   "Pick a held-out procedure in Step 1 for true gold grading.")
    step_ui.render_trajectory(ss.rollout, ss.branches or [], ss.has_gold)


# ---- human-in-the-loop labeling of the agent's decisions (process supervision)
if ss.rollout:
    with st.expander("✍️ Label the agent's decisions (process supervision)"):
        st.caption("Mark each step's picked action correct or wrong. Pre-filled from the "
                   "automatic on-path / off-path check — just correct where you disagree.")
        _steps = ss.rollout.get("steps") or []
        _rkp = f"roll_{ss.source_name}_{ss.file_index}"
        with st.form(_rkp):
            _step_labels = []
            for _s in _steps:
                _auto = _s.get("is_valid", False)
                _s1, _s2, _s3 = st.columns([1, 4, 3])
                _s1.markdown(f"**#{_s['step']}**")
                _s2.markdown(f"picked: `{_s.get('picked')}`")
                _s2.caption("auto: ✅ on-path" if _auto else "auto: ❌ off-path")
                _hl = _s3.radio("label", ["correct", "wrong"], index=0 if _auto else 1,
                                horizontal=True, key=f"sl_{_rkp}_{_s['step']}",
                                label_visibility="collapsed")
                _step_labels.append({"step": _s["step"], "picked": _s.get("picked"),
                                     "auto_is_valid": _auto, "human_label": _hl})
            _rnotes = st.text_area("Notes (optional)", key=f"rn_{_rkp}")
            _rsaved = st.form_submit_button("💾 Save decision labels", type="primary")
        if _rsaved:
            ANN.append_record({
                "kind": "rollout_labels",
                "source": ss.source_name,
                "file_index": ss.file_index,
                "has_gold": ss.has_gold,
                "run_label": ss.run_label,
                "procedure_text": ss.procedure_text,
                "step_labels": _step_labels,
                "completed_trajectory": ss.rollout.get("completed_trajectory"),
            })
            st.success(f"Saved · {len(_step_labels)} step decisions labeled. Dataset updated below.")

# =================================================================== DATASET
st.divider()
st.header("Labeled dataset · data flywheel")
st.caption("Every save above is appended to demo/annotations/labeled_data.jsonl — a growing, "
           "retrain-ready corpus built with near-zero manual effort.")
_sm = ANN.summary()
_d1, _d2, _d3, _d4, _d5 = st.columns(5)
_d1.metric("Records", _sm["records"])
_d2.metric("Extraction recs", _sm["extraction_records"])
_d3.metric("Action labels", _sm["action_labels"])
_d4.metric("Rollout recs", _sm["rollout_records"])
_d5.metric("Step labels", _sm["step_labels"])
_bytes = ANN.raw_bytes()
st.download_button("⬇️ Download dataset (JSONL)", data=_bytes,
                   file_name="labeled_data.jsonl", mime="text/plain",
                   disabled=not _bytes)
if _bytes:
    with st.expander("Recent saved records"):
        for _r in ANN.load_all()[-10:][::-1]:
            st.markdown(f"- `{_r.get('timestamp')}` · **{_r.get('kind')}** · "
                        f"source: {_r.get('source')} · file_index: {_r.get('file_index')}")
