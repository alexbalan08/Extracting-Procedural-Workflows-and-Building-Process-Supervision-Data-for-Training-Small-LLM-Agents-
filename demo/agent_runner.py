"""Step 4/5 wrapper: drive the planning agents and grade their trajectory.

Reuses the project's own runner.walk_free and Agent_testing/agents.py classes so
the demo runs the EXACT inference path the thesis evaluates. Two execution modes,
both producing the same rollout shape (rollout["steps"] with per-candidate
prm_dist/llm_dist/final and tool info), so one renderer covers both:

  * live  — instantiate the chosen config and run walk_free (needs the GPU box
            for M3/M4; OpenAI configs need only an API key)
  * replay — load a saved Agent_testing/results/inference_*.json record

Grading (Step 5) is always against the gold graph when available; for a brand-new
PDF with no gold we grade against the extracted graph itself (clearly labeled).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "Agent_testing"
RESULTS_DIR = AGENT_DIR / "results"

# runner.py adds Extraction_results to sys.path itself; we just need Agent_testing.
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ----------------------------------------------------------------- config menu
# label -> (factory-key, needs_candidates, needs_gpu, needs_openai)
CONFIGS: dict[str, dict] = {
    "M1 · Llama bare (no candidates)":        {"key": "llama_bare",     "cands": False, "gpu": True,  "openai": False},
    "M2 · Llama + actions":                   {"key": "llama_actions",  "cands": True,  "gpu": True,  "openai": False},
    "M3 · Ensemble (Llama + PRM blend)":      {"key": "ensemble",       "cands": True,  "gpu": True,  "openai": False},
    "M4 · Agentic ensemble (+ graph tool)":   {"key": "agentic",        "cands": True,  "gpu": True,  "openai": False},
    "OpenAI bare (baseline)":                 {"key": "openai_bare",    "cands": False, "gpu": False, "openai": True},
    "OpenAI + actions (baseline)":            {"key": "openai_actions", "cands": True,  "gpu": False, "openai": True},
}


def make_agent(key: str, *, alpha: float = 0.90, tool_threshold: float = 0.45,
               tool_margin: float = 0.20, openai_model: str = "gpt-5.4-mini"):
    import agents as A
    if key == "llama_bare":
        return A.LlamaBareAgent()
    if key == "llama_actions":
        return A.LlamaActionsAgent()
    if key == "ensemble":
        return A.EnsemblePlannerAgent(alpha=alpha)
    if key == "agentic":
        return A.AgenticEnsembleAgent(alpha=alpha, tool_threshold=tool_threshold, tool_margin=tool_margin)
    if key == "openai_bare":
        return A.OpenAIBareAgent(model=openai_model)
    if key == "openai_actions":
        return A.OpenAIActionsAgent(model=openai_model)
    raise ValueError(f"unknown config key: {key}")


# ------------------------------------------------------------------- case build
def build_case(extraction_result: dict, procedure_text: str,
               file_index, held_out_record: dict | None):
    """Build a runner.ProcedureCase from a Step-2 extraction.

    If held_out_record is given we get true gold (fuzzy-matched to the prediction).
    Otherwise we grade against the extracted graph itself (gold := predicted)."""
    from runner import ProcedureCase

    pred_wf = extraction_result.get("workflow") or {}
    pred_actions = pred_wf.get("actions") or []
    pred_states = extraction_result.get("execution_states") or []

    if held_out_record is not None:
        # reuse the real builder — does match_actions fuzzy alignment pred<->gold
        prediction_record = {"workflow": pred_wf, "execution_states": pred_states}
        case = ProcedureCase.build(held_out_record, prediction_record)
        case.procedure_text = procedure_text
        return case, True

    # no gold: self-graph grading. gold fields mirror predicted; name->id identity map.
    id_to_name = {a["id"]: a["name"] for a in pred_actions}
    name_to_id = {a["name"]: a["id"] for a in pred_actions}
    case = ProcedureCase(
        file_index=file_index if file_index is not None else -1,
        procedure_text=procedure_text,
        pred_action_names=[a["name"] for a in pred_actions],
        pred_id_to_name=id_to_name,
        pred_execution_states=pred_states,
        gold_action_names=[a["name"] for a in pred_actions],
        gold_id_to_name=id_to_name,
        gold_execution_states=pred_states,
        pred_name_to_gold_id=name_to_id,
    )
    return case, False


def gold_branches(case) -> list[dict]:
    """Enumerated reference paths (gold if available, else the extracted graph),
    as action-name lists — the green/red reference for Step 5."""
    from runner import enumerate_paths
    paths = enumerate_paths(case.gold_execution_states)
    return [{"branch": i, "path": [case.gold_id_to_name.get(a, a) for a in p]}
            for i, p in enumerate(paths)]


def run_live(case, agent, *, mode: str = "predicted", give_candidates: bool = True,
             max_steps: int = 20) -> dict:
    """Run the chosen agent and return the rollout dict (same shape as saved runs)."""
    from runner import walk_free
    return walk_free(case, agent, max_steps=max_steps,
                     give_candidates=give_candidates, mode=mode)


# ----------------------------------------------------------------------- replay
def list_result_files() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.glob("inference_*.json"))


def load_replay(filename: str, file_index) -> dict | None:
    """Return the saved record (with rollout + branches) for file_index, or None."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if r.get("file_index") == file_index:
            return r
    return None
