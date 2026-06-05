#stratifies the main 4-method comparison by procedure complexity to answer
#"do complex procedures fail more?". two complexity dimensions, each split into
#three tertile bins (low / mid / high) over the 49 held-out procedures:
#  - # of actions in the gold workflow (candidate-set size)
#  - # of gateways in the gold workflow (decision-point count)
#we reuse the same per-method-and-graph filter as evaluate_traces.py so this
#table covers exactly the canonical 4 methods x 2 graphs at big PRM, alpha=0.9.
#
#  python evaluate_complexity.py


import json
from pathlib import Path

from evaluate_traces import (
    METHODS,
    ENSEMBLE_CANONICAL_SUFFIX,
    SMALL_PRM_MARKER,
    _compute_metrics,
)
from eval_common import load_actions_lookup, parse_method, print_table, write_csv


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"


#per-procedure complexity from the gold workflow — the "true" complexity, independent of extraction
def _load_complexity(held_out_path: Path) -> dict[int, dict[str, int]]:
    with open(held_out_path, encoding="utf-8") as f:
        held_out = json.load(f)
    out = {}
    for rec in held_out:
        wf = rec.get("workflow") or {}
        out[rec["file_index"]] = {
            "n_actions":  len(wf.get("actions", [])),
            "n_gateways": len(wf.get("gateways", [])),
        }
    return out


#tertile cutoffs over the actual distribution so the three bins hold roughly equal n
def _tertile_cutoffs(values: list[int]) -> tuple[int, int]:
    sv = sorted(values)
    n = len(sv)
    return sv[n // 3], sv[2 * n // 3]


def _bin_of(v: int, low_max: int, mid_max: int) -> str:
    if v <= low_max:
        return "low"
    if v <= mid_max:
        return "mid"
    return "high"


def main():
    actions_lookup = load_actions_lookup(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)

    complexity = _load_complexity(DEFAULT_HELD_OUT)

    #collect canonical inference files — same filters as evaluate_traces.py
    runs = []
    for path in sorted(RESULTS_DIR.glob("inference_*.json")):
        method = parse_method(path.name, METHODS)
        if method is None:
            continue
        if path.name.endswith(SMALL_PRM_MARKER):
            continue
        if method == "ensemble" and ENSEMBLE_CANONICAL_SUFFIX not in path.name:
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces or len(traces) != 49:
            continue
        runs.append((method, traces[0]["eval_mode"], traces))

    if not runs:
        print("No canonical inference files found.")
        return

    #tertile cutoffs computed once over the held-out set the inference files actually cover
    file_indices = [tr["file_index"] for tr in runs[0][2]]
    act_vals = [complexity[fi]["n_actions"]  for fi in file_indices if fi in complexity]
    gw_vals  = [complexity[fi]["n_gateways"] for fi in file_indices if fi in complexity]
    act_low, act_mid = _tertile_cutoffs(act_vals)
    gw_low,  gw_mid  = _tertile_cutoffs(gw_vals)

    print("Tertile cutoffs (over the 49 held-out procedures):")
    print(f"  # actions  : low ≤ {act_low}, mid ≤ {act_mid}, high > {act_mid}")
    print(f"  # gateways : low ≤ {gw_low}, mid ≤ {gw_mid}, high > {gw_mid}")

    rows_actions = _stratify(runs, complexity, "n_actions",  act_low, act_mid, actions_lookup)
    rows_gateways = _stratify(runs, complexity, "n_gateways", gw_low,  gw_mid,  actions_lookup)

    for rows, fname in [(rows_actions, "complexity_actions.csv"),
                        (rows_gateways, "complexity_gateways.csv")]:
        write_csv(RESULTS_DIR / fname, rows)

    table_keys = ["method", "graph", "bin", "n", "step_valid_%", "completed_%"]
    print()
    print(f"=== Stratified by # of actions  (low ≤ {act_low}, mid ≤ {act_mid}, high > {act_mid}) ===")
    print_table(rows_actions, table_keys)
    print()
    print(f"=== Stratified by # of gateways (low ≤ {gw_low}, mid ≤ {gw_mid}, high > {gw_mid}) ===")
    print_table(rows_gateways, table_keys)
    print(f"\nWrote complexity_actions.csv and complexity_gateways.csv to {RESULTS_DIR}")


def _stratify(runs, complexity, dim_name, low_max, mid_max, actions_lookup):
    bin_order = {"low": 0, "mid": 1, "high": 2}
    rows = []
    for method, graph, traces in runs:
        bins: dict[str, list] = {"low": [], "mid": [], "high": []}
        for tr in traces:
            fi = tr["file_index"]
            if fi not in complexity:
                continue
            bins[_bin_of(complexity[fi][dim_name], low_max, mid_max)].append(tr)
        for bin_name, bin_traces in bins.items():
            if not bin_traces:
                continue
            m = _compute_metrics(bin_traces, actions_lookup)
            rows.append({
                "method":       method,
                "graph":        graph,
                "bin":          bin_name,
                "n":            len(bin_traces),
                "step_valid_%": m["step_valid_%"],
                "completed_%":  m["completed_%"],
            })
    rows.sort(key=lambda r: (METHODS.index(r["method"]),
                             0 if r["graph"] == "predicted" else 1,
                             bin_order[r["bin"]]))
    return rows


if __name__ == "__main__":
    main()
