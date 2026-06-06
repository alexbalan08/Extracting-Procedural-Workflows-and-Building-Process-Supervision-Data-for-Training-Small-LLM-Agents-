

import contextlib
import csv
import io
from pathlib import Path

from runner import load_cases



def load_actions_lookup(held_out_path: Path, predictions_path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(held_out_path, predictions_path)
    lookup = {}
    for c in cases:
        lookup[(c.file_index, "predicted")] = c.pred_action_names
        lookup[(c.file_index, "gold")]      = c.gold_action_names
    return lookup



def parse_method(filename: str, methods: list[str]) -> str | None:
    base = filename[len("inference_"):]
    for m in methods:
        if base.startswith(m + "_"):
            return m
    return None


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, float) else str(v)



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
