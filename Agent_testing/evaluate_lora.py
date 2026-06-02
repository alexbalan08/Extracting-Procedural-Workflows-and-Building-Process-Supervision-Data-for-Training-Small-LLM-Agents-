#dedicated comparison table for the two LoRA adapters (big = trained_model,
#small = trained_model_small). only shows PRM-using methods — ensemble and
#agentic_ensemble — because the non-PRM baselines (llama_bare, llama_actions)
#don't depend on the adapter and so there's nothing to compare for them.
#
#  python evaluate_lora.py
#
#reads inference_*.json files produced by lora_configs.py. each row carries a
#"prm" column showing which adapter the agent used (big / small).

import contextlib
import csv
import io
import json
from pathlib import Path

from runner import load_cases
from evaluate_traces import _compute_metrics


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"

#longest first so "agentic_ensemble" matches before "ensemble".
#non-PRM methods (llama_bare, llama_actions) are deliberately excluded — the LoRA
#choice can't affect them, so they're not part of this comparison.
METHODS = ["agentic_ensemble", "ensemble"]

#filename suffixes that count as the canonical config for each method
ENSEMBLE_BIG_SUFFIX   = "_alpha0.90.json"
ENSEMBLE_SMALL_SUFFIX = "_alpha0.90_small.json"
AGENTIC_BIG_SUFFIX    = "_alpha0.90_t0.45_m0.20.json"
AGENTIC_SMALL_SUFFIX  = "_alpha0.90_t0.45_m0.20_small.json"


def _parse_method(filename: str) -> str | None:
    base = filename[len("inference_"):]
    for m in METHODS:
        if base.startswith(m + "_"):
            return m
    return None


def _is_canonical(method: str, filename: str) -> bool:
    if method == "ensemble":
        return filename.endswith(ENSEMBLE_BIG_SUFFIX) or filename.endswith(ENSEMBLE_SMALL_SUFFIX)
    if method == "agentic_ensemble":
        return filename.endswith(AGENTIC_BIG_SUFFIX) or filename.endswith(AGENTIC_SMALL_SUFFIX)
    return False


def _parse_prm_variant(filename: str) -> str:
    #"_small.json" filename suffix marks the dedup-data PRM; anything else is big.
    return "small" if filename.endswith("_small.json") else "big"


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)
    actions_lookup = {}
    for c in cases:
        actions_lookup[(c.file_index, "predicted")] = c.pred_action_names
        actions_lookup[(c.file_index, "gold")]      = c.gold_action_names

    rows = []
    for path in sorted(RESULTS_DIR.glob("inference_*.json")):
        method = _parse_method(path.name)
        if method is None or not _is_canonical(method, path.name):
            continue
        with open(path, encoding="utf-8") as f:
            traces = json.load(f)
        if not traces:
            continue
        graph = traces[0].get("eval_mode", "predicted")
        rows.append({
            "method": method,
            "graph":  graph,
            "prm":    _parse_prm_variant(path.name),
            "file":   path.name,
            **_compute_metrics(traces, actions_lookup),
        })

    if not rows:
        print(f"No canonical ensemble / agentic_ensemble files found in {RESULTS_DIR}")
        return

    #group by method → graph → PRM. big before small so the row-to-row delta tells
    #you what the small PRM cost or gained within each (method, graph) pair.
    _prm_order = {"big": 0, "small": 1}
    rows.sort(key=lambda r: (
        METHODS.index(r["method"]),
        0 if r["graph"] == "predicted" else 1,
        _prm_order.get(r["prm"], 2),
    ))

    out_csv = RESULTS_DIR / "lora_compare.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    keys = ["method", "graph", "prm", "procedures",
            "step_valid_%", "first_step_valid_%", "completed_%",
            "avg_steps_to_offpath", "tool_fired_%"]
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
