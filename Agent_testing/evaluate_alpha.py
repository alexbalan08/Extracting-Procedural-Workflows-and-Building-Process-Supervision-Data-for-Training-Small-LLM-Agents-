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
    #pulls "0.90" out of "inference_ensemble_<graph>_alpha0.90.json"
    m = re.search(r"_alpha(\d+\.\d+)", filename)
    return m.group(1) if m else ""


def main():
    #load_cases prints a "Loaded N cases" line we don't need here, swallow it
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)
    actions_lookup = {}
    for c in cases:
        actions_lookup[(c.file_index, "predicted")] = c.pred_action_names
        actions_lookup[(c.file_index, "gold")]      = c.gold_action_names

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_ensemble_*.json")):
        #skip agentic_ensemble — its filename also starts with "inference_ensemble" once
        #we strip the prefix differently. easiest: explicit exclusion.
        if path.name.startswith("inference_agentic_ensemble"):
            continue
        alpha_str = _parse_alpha(path.name)
        if not alpha_str:
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces:
            continue
        #the alpha sweep was only run on --limit 5, so we filter to N=5 here so the
        #table is consistent. mixing N=5 sweep rows with the existing N=49 alpha=0.9
        #file would make the curve misleading.
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

    #predicted before gold, then alpha ascending — sweep reads naturally top-to-bottom
    rows.sort(key=lambda r: (0 if r["graph"] == "predicted" else 1, float(r["alpha"])))

    out_csv = RESULTS_DIR / "alpha_sweep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    
    #only the columns we need for the writeup figure — alpha sweep is currently
    #predicted-only so we drop the graph column too. add it back if you ever sweep on gold.
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
