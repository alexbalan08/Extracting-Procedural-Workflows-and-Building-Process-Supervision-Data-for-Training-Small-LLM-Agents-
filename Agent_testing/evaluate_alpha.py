#we have this to motivate the alpha choice in the writeup from some automatic validation
#but i also did manual validation and identified the types of errors by looking at the probbailtiites of 
#the next actions and decided 0.9 is the best choice




import contextlib
import csv
import io
import json
import re
from pathlib import Path

from runner import load_cases
from evaluate_traces import _compute_metrics


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"


def _parse_alpha(filename: str) -> str:
    m = re.search(r"_alpha(\d+\.\d+)", filename)
    return m.group(1) if m else ""


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)
    actions_lookup = {}
    for c in cases:
        actions_lookup[(c.file_index, "predicted")] = c.pred_action_names
        actions_lookup[(c.file_index, "gold")]      = c.gold_action_names

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_ensemble_*.json")):
        if path.name.startswith("inference_agentic_ensemble"):
            continue
        alpha_str = _parse_alpha(path.name)
        if not alpha_str:
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces:
            continue

        if len(traces) != 5:
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
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    
    #only the columns we need
    keys = ["alpha", "step_valid_%", "completed_%"]
    widths = [max(len(k), max(len(_fmt(r.get(k, ""))) for r in rows)) for k in keys]

    line = " | ".join(k.ljust(w) for k, w in zip(keys, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(_fmt(r[k]).ljust(w) for k, w in zip(keys, widths)))
    print(f"\nWrote {out_csv}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


if __name__ == "__main__":
    main()
