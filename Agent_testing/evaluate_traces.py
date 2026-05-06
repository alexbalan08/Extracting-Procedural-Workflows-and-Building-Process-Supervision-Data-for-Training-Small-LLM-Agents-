#reads the inference_*.json files in results/ and prints per-method-and-graph metrics
#the gap between a method's "predicted" row and its "gold" row tells us how much extraction
#noise costs that method. invented/snapped only matter for llama_bare since the others get
#the candidate list — invented should be ~0 for them.

#  python evaluate_traces.py


import csv
import json
from pathlib import Path

from runner import load_cases


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"

#longest first so "agentic_ensemble" matches before "ensemble"
METHODS = ["agentic_ensemble", "ensemble", "llama_actions", "llama_bare"]


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


def _parse_method(filename: str) -> str | None:
    base = filename[len("inference_"):]
    for m in METHODS:
        if base.startswith(m + "_"):
            return m
    return None


#aggregates one inference file into the metric row we print
def _compute_metrics(traces: list[dict], actions_lookup: dict) -> dict:
    n_proc = len(traces)
    total_steps = valid_steps = 0
    first_correct = first_total = 0
    completed = exact_match = 0
    off_path_step_counts = []
    invented = snapped = 0
    tool_fired = tool_useful = 0

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
            cls = _classify_pick(s["picked"], actions)
            if cls == "invented":
                invented += 1
            elif cls == "snapped":
                snapped += 1
            if s.get("tool_called"):
                tool_fired += 1
                if (s.get("tool") or {}).get("tool_useful"):
                    tool_useful += 1

        if rollout.get("status") == "completed":
            completed += 1
        if rollout.get("match_type") == "exact":
            exact_match += 1
        if rollout.get("status") == "off_path":
            off_path_step_counts.append(len(steps))

    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "procedures":           n_proc,
        "step_valid_%":         pct(valid_steps, total_steps),
        "first_step_valid_%":   pct(first_correct, first_total),
        "completed_%":          pct(completed, n_proc),
        "exact_branch_%":       pct(exact_match, n_proc),
        "avg_steps_to_offpath": avg(off_path_step_counts),
        "invented_%":           pct(invented, total_steps),
        "snapped_%":            pct(snapped, total_steps),
        "tool_fired_%":         pct(tool_fired, total_steps),
        "tool_useful_%":        pct(tool_useful, tool_fired) if tool_fired else 0.0,
    }


def main():
    cases = load_cases(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)
    #cache action names per (file_index, mode) so llama_bare (no candidate_actions stored) works
    actions_lookup = {}
    for c in cases:
        actions_lookup[(c.file_index, "predicted")] = c.pred_action_names
        actions_lookup[(c.file_index, "gold")]      = c.gold_action_names

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_*.json")):
        method = _parse_method(path.name)
        if method is None:
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

    #group by method, predicted before gold so the gap reads top-to-bottom per pair
    rows.sort(key=lambda r: (METHODS.index(r["method"]), 0 if r["graph"] == "predicted" else 1))

    out_csv = RESULTS_DIR / "metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    #full metric names so the table reads on its own. tool_* columns are 0 outside agentic_ensemble.
    keys = ["method", "graph", "procedures",
            "step_valid_%", "first_step_valid_%", "completed_%", "exact_branch_%",
            "avg_steps_to_offpath", "invented_%", "snapped_%",
            "tool_fired_%", "tool_useful_%"]
    widths = [max(len(k), max(len(_fmt(r.get(k, ""))) for r in rows)) for k in keys]

    line = " | ".join(k.ljust(w) for k, w in zip(keys, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(_fmt(r[k]).ljust(w) for k, w in zip(keys, widths)))
    print(_legend())
    print(f"\nWrote {out_csv}")


#one-liner per column so anyone reading the printout knows what each number means
def _legend() -> str:
    return (
        "\n"
        "step_valid_%         = % of steps where the pick was a valid next action in the active graph\n"
        "first_step_valid_%   = same metric but only on step 1 (no compounding error)\n"
        "completed_%          = % of procedures the agent walked all the way to a terminal state\n"
        "exact_branch_%       = % where the full trajectory equals one enumerated path through the graph\n"
        "avg_steps_to_offpath = avg steps taken before going off-path (failures only)\n"
        "invented_%           = % of picks the fuzzy matcher could not snap to any real action (mainly llama_bare)\n"
        "snapped_%            = % of picks that were not exact strings but were fuzzy-matched to an action\n"
        "tool_fired_%         = % of steps where the agentic gate consulted the graph (agentic_ensemble only)\n"
        "tool_useful_%        = of the firings, % where the graph actually narrowed the candidate set"
    )


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


if __name__ == "__main__":
    main()
