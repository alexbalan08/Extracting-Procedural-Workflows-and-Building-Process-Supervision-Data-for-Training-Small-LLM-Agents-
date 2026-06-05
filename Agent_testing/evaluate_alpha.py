#we have this to motivate the alpha choice in the writeup from some automatic validation
#but i also did manual validation and identified the types of errors by looking at the probbailtiites of 
#the next actions and decided 0.9 is the best choice




import json
import re
from pathlib import Path

from evaluate_traces import _compute_metrics
from eval_common import load_actions_lookup, print_table, write_csv


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"


def _parse_alpha(filename: str) -> str:
    m = re.search(r"_alpha(\d+\.\d+)", filename)
    return m.group(1) if m else ""


def main():
    actions_lookup = load_actions_lookup(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_ensemble_*.json")):
        if path.name.startswith("inference_agentic_ensemble"):
            continue
        #the small-PRM variant is its own comparison (evaluate_lora.py); skip so it doesn't
        #show up as a second alpha=0.90 row here
        if path.name.endswith("_small.json"):
            continue
        alpha_str = _parse_alpha(path.name)
        if not alpha_str:
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces:
            continue

        graph = traces[0].get("eval_mode", "predicted")
        rows.append({
            "graph": graph,
            "alpha": alpha_str,
            "file": path.name,
            **_compute_metrics(traces, actions_lookup),
        })

    if not rows:
        print(f"No inference_ensemble_*.json files found in {RESULTS_DIR}")
        return

    rows.sort(key=lambda r: (0 if r["graph"] == "predicted" else 1, float(r["alpha"])))

    out_csv = RESULTS_DIR / "alpha_sweep.csv"
    write_csv(out_csv, rows)

    #"procedures" makes the sample size explicit — the quick sweep ran on fewer procedures than
    #the canonical alpha=0.90 point, so the rows are not all measured over the same n
    print_table(rows, ["alpha", "graph", "procedures", "step_valid_%", "completed_%"])
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
