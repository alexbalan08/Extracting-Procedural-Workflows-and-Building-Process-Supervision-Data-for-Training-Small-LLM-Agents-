#shared helpers for the evaluate_*.py report scripts: building the action-name
#lookup the metrics need, parsing method names off inference filenames, and
#printing/writing the metric tables. keeps the four report scripts to just their
#own filtering and column choices.

import contextlib
import csv
import io
from pathlib import Path

from runner import load_cases


#cache action names per (file_index, mode) so llama_bare (which stores no
#candidate_actions) can fall back to the active graph's actions when scoring.
#load_cases prints a "Loaded N cases" line we don't want in a report, so swallow it.
def load_actions_lookup(held_out_path: Path, predictions_path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(held_out_path, predictions_path)
    lookup = {}
    for c in cases:
        lookup[(c.file_index, "predicted")] = c.pred_action_names
        lookup[(c.file_index, "gold")]      = c.gold_action_names
    return lookup


#strip the "inference_" prefix and match the longest method name that fits.
#methods must be ordered longest-first so "agentic_ensemble" wins over "ensemble".
def parse_method(filename: str, methods: list[str]) -> str | None:
    base = filename[len("inference_"):]
    for m in methods:
        if base.startswith(m + "_"):
            return m
    return None


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, float) else str(v)


#prints an aligned " | "-separated table for the given column keys
def print_table(rows: list[dict], keys: list[str]) -> None:
    widths = [max(len(k), max(len(fmt(r.get(k, ""))) for r in rows)) for k in keys]
    header = " | ".join(k.ljust(w) for k, w in zip(keys, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(fmt(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
