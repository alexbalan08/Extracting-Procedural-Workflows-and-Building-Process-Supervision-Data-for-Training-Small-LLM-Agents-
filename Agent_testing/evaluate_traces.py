#reads the inference_*.json files in results/ and prints per-method-and-graph metrics.
#the gap between a method's "predicted" row (agent sees predicted graph) and its "gold" row
#(agent sees gold graph) tells us how much extraction noise costs that method —
#both rows are validated against ground truth.

#  python evaluate_traces.py


import json
from pathlib import Path

from eval_common import load_actions_lookup, parse_method, print_table, write_csv


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"

DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"

#longest first so "agentic_ensemble" matches before "ensemble"
METHODS = ["agentic_ensemble", "ensemble",
           "llama_actions", "openai_actions",
           "llama_bare",    "openai_bare"]

#only the canonical config of each method shows up in the main table.
#ensemble alpha sweeps live in evaluate_alpha.py; big-vs-small PRM comparison lives in
#evaluate_lora.py. here we keep only the canonical big-PRM run at alpha=0.90.
ENSEMBLE_CANONICAL_SUFFIX = "_alpha0.90.json"
SMALL_PRM_MARKER = "_small.json"


#mirror runner._resolve_picked_id rules so we know whether a pick was already a real action,
#fuzzy-matched to one (snapped), or unmatchable (invented). only meaningful for llama_bare.
def _classify_pick(picked: str, action_names: list[str]) -> str:
    if picked in action_names:
        return "exact"
    pl = picked.lower().strip().rstrip(".")
    for name in action_names:
        nl = name.lower()
        if nl == pl or nl in pl or pl.startswith(nl):
            return "snapped"
    return "invented"


#aggregates one inference file into the metric row we print
def _compute_metrics(traces: list[dict], actions_lookup: dict) -> dict:
    n_proc = len(traces)
    total_steps = valid_steps = 0
    first_correct = first_total = 0
    completed = 0
    off_path_step_counts = []
    invented = 0
    tool_fired = 0

    for tr in traces:
        rollout = tr["rollout"]
        steps = rollout.get("steps", [])
        #candidate_actions is None for llama_bare, fall back to active-graph actions
        actions = tr.get("candidate_actions") or actions_lookup[(tr["file_index"], tr["eval_mode"])]

        for i, s in enumerate(steps):
            total_steps += 1
            if s.get("is_valid"):
                valid_steps += 1
            if i == 0:
                first_total += 1
                if s.get("is_valid"):
                    first_correct += 1
            if _classify_pick(s["picked"], actions) == "invented":
                invented += 1
            if s.get("tool_called"):
                tool_fired += 1

        if rollout.get("status") == "completed":
            completed += 1
        if rollout.get("status") == "off_path":
            #the failing step is appended before break, so subtract it to count only the valid prefix
            off_path_step_counts.append(max(len(steps) - 1, 0))

    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "procedures":           n_proc,
        "step_valid_%":         pct(valid_steps, total_steps),
        "first_step_valid_%":   pct(first_correct, first_total),
        "completed_%":          pct(completed, n_proc),
        "avg_steps_to_offpath": avg(off_path_step_counts),
        "invented_%":           pct(invented, total_steps),
        "tool_fired_%":         pct(tool_fired, total_steps),
    }


def main():
    actions_lookup = load_actions_lookup(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_*.json")):
        method = parse_method(path.name, METHODS)
        if method is None:
            continue
        #small-PRM runs are reported in evaluate_lora.py, drop them from the main table
        if path.name.endswith(SMALL_PRM_MARKER):
            continue
        #ensemble alpha sweep is reported separately — only keep the canonical alpha here
        if method == "ensemble" and ENSEMBLE_CANONICAL_SUFFIX not in path.name:
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces:
            continue
        graph = traces[0].get("eval_mode", "predicted")
        rows.append({
            "method": method, "graph": graph, "file": path.name,
            **_compute_metrics(traces, actions_lookup),
        })

    if not rows:
        print(f"No inference_*.json files found in {RESULTS_DIR}")
        return

    #group by method, predicted before gold so the extraction-cost gap reads top-to-bottom per pair
    rows.sort(key=lambda r: (METHODS.index(r["method"]), 0 if r["graph"] == "predicted" else 1))

    out_csv = RESULTS_DIR / "metrics.csv"
    write_csv(out_csv, rows)

    #full metric names so the table reads on its own. tool_* columns are 0 outside agentic_ensemble.
    keys = ["method", "graph", "procedures",
            "step_valid_%", "first_step_valid_%", "completed_%",
            "avg_steps_to_offpath", "invented_%",
            "tool_fired_%"]
    print_table(rows, keys)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
