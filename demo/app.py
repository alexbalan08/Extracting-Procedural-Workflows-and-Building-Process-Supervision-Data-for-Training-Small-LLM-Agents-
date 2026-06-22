#this is out streamlit demo, mostly code generated with Claude Code since this is basic streamlit functionality 



from __future__ import annotations

import json
import os

import streamlit as st

import agent_runner as AR
import extraction as X
import step_ui
from pdf_utils import extract_text
from viz import workflow_to_dot

st.set_page_config(page_title="Procedural Workflow Pipeline — Demo", layout="wide")


ss = st.session_state
ss.setdefault("procedure_text", "")
ss.setdefault("pdf_meta", None)
ss.setdefault("source_name", None)
ss.setdefault("file_index", None)      # set when the text is a held-out procedure gold available
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
        "5. Visualize trajectory"
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
    st.caption("Needs a CUDA GPU and valid OpenAI key.")

# =================================================================== STEP 1
st.header("Step 1 · Get the procedure text")

source = st.radio(
    "Source", ["Upload PDF", "Paste text", "Pick held-out procedure (has gold)"],
    horizontal=True, label_visibility="collapsed",
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
        choice = st.selectbox("Held-out procedure", list(labels.keys()))
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
    st.caption("Runs the real pipeline: 5-shot prompt → structural checker → LLM semantic checker → self-refine loop"
    ".")

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
                 "well-formed, edges point to real nodes. Catches wrong JSON files before "
                 "the more expensive LLM critic runs.",
        )

    go_live = st.button("Run extraction (live process)", type="primary")

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

# =================================================================== STEP 4
if ss.workflow:
    st.divider()
    st.header("Step 4 · Run the planning agent step by step")

    exec_mode = st.radio("Execution", ["Live inference", "Replay saved run"], horizontal=True)

    if exec_mode == "Live inference":
        cfg_label = st.selectbox("Agent configuration", list(AR.CONFIGS.keys()),
                                 index=list(AR.CONFIGS.keys()).index("M4 · Agentic ensemble (+ graph tool)"))
        cfg = AR.CONFIGS[cfg_label]

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            grade_mode = st.selectbox(
                "Graph the agent sees", ["predicted", "gold"],
                help="predicted = the extracted graph (real deployment). gold needs a held-out procedure.",
            )
            if grade_mode == "gold" and ss.file_index is None:
                st.caption("⚠️ gold needs a held-out procedure; will use predicted.")
                grade_mode = "predicted"
        with p2:
            alpha = st.slider("α (PRM weight)", 0.0, 1.0, 0.90, 0.05,
                              disabled=cfg["key"] not in ("ensemble", "agentic"))
        with p3:
            tool_threshold = st.slider("tool threshold", 0.0, 1.0, 0.45, 0.05,
                                       disabled=cfg["key"] != "agentic")
        with p4:
            tool_margin = st.slider("tool margin", 0.0, 1.0, 0.20, 0.05,
                                    disabled=cfg["key"] != "agentic")
        openai_model_agent = st.text_input("OpenAI model (for OpenAI configs)", value="gpt-5.4-mini") \
            if cfg["openai"] else "gpt-5.4-mini"

        if cfg["gpu"]:
            st.info("ℹ️ This config loads Llama-3.1-8B (4-bit) + PRM LoRA — runs on your GPU with CUDA, "
                    "not on the Mac.")

        if st.button("Run agent", type="primary"):
            try:
                with st.spinner(f"Running {cfg_label}…"):
                    held = X.held_out_record(ss.file_index) if ss.file_index is not None else None
                    case, has_gold = AR.build_case(ss.extraction, ss.procedure_text, ss.file_index, held)
                    #keep at most ONE gpu-resident agent across streamlit reruns. clicking the
                    #same gpu config reuses it (no reload); switching to a different gpu config
                    #evicts the previous one and frees its VRAM first, so two 4-bit base models
                    #never coexist on the GPU (that double-load was the OOM source).
                    cache_key = (cfg["key"], alpha, tool_threshold, tool_margin, openai_model_agent)
                    if cfg["gpu"]:
                        if ss.get("gpu_agent_key") != cache_key:
                            AR.free_agent(ss.get("gpu_agent"))
                            ss.gpu_agent = AR.make_agent(
                                cfg["key"], alpha=alpha, tool_threshold=tool_threshold,
                                tool_margin=tool_margin, openai_model=openai_model_agent)
                            ss.gpu_agent_key = cache_key
                        agent = ss.gpu_agent
                    else:
                        #openai agents hold no GPU memory — build fresh, nothing to cache/evict
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
            rfile = st.selectbox("Saved run", files, index=default_i)
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
            #selectbox works with any step count (including 1); st.slider requires min<max
            #clamp ss.step_ptr in case the previous rollout had more steps than this one
            ss.step_ptr = min(ss.step_ptr, len(steps) - 1)
            ss.step_ptr = st.selectbox(
                "Step", options=list(range(len(steps))),
                format_func=lambda i: f"{i + 1} / {len(steps)}",
                index=ss.step_ptr,
            )
        with nav4:
            show_all = st.toggle("Show all steps", value=False)

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
