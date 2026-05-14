#sanity-checks the PRM training data and prints a short report.
#runs three passes:
#  1. step-level class balance (Yes vs No after expansion)
#  2. corruption-type distribution
#  3. near-duplicate traces (same procedure + identical action sequence)
#
#  python validate_training_data.py


import json
from collections import Counter
from pathlib import Path


_HERE = Path(__file__).parent
TRACES_PATH = _HERE / "prm_training_data.json"
SFT_PATH = _HERE / "prm_sft_train.jsonl"


def _load_traces() -> list[dict]:
    with open(TRACES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_sft() -> list[dict]:
    records = []
    with open(SFT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _bar(n: int, total: int, width: int = 30) -> str:
    #ascii so windows cp1252 console doesn't choke
    if total <= 0:
        return ""
    filled = round(width * n / total)
    return "#" * filled + "." * (width - filled)


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def step_level_balance(sft_records: list[dict]) -> None:
    yes = sum(1 for r in sft_records if r["messages"][-1]["content"].startswith("Yes"))
    no = sum(1 for r in sft_records if r["messages"][-1]["content"].startswith("No"))
    other = len(sft_records) - yes - no
    print(f"\n--- 1. Step-level class balance ({len(sft_records)} SFT examples) ---")
    print(f"  Yes : {yes:5d}  {_pct(yes, len(sft_records))}  {_bar(yes, len(sft_records))}")
    print(f"  No  : {no:5d}  {_pct(no, len(sft_records))}  {_bar(no, len(sft_records))}")
    if other:
        print(f"  ??? : {other:5d}  {_pct(other, len(sft_records))}")
    ratio = no / yes if yes else 0
    print(f"  No : Yes ratio = 1 : {ratio:.2f}")


def corruption_distribution(traces: list[dict]) -> None:
    types = Counter(t.get("corruption_type") for t in traces)
    print(f"\n--- 2. Corruption-type distribution ---")
    for ctype, n in types.most_common():
        label = ctype if ctype is not None else "(none — positive trace)"
        print(f"  {label:25s} : {n:5d}  {_pct(n, len(traces))}  {_bar(n, len(traces))}")


def near_duplicates(traces: list[dict]) -> None:
    #key on (file_index, action sequence) — two traces with identical actions on the
    #same procedure are duplicates regardless of labels or corruption metadata
    keys = Counter()
    for t in traces:
        seq = tuple(s["action"] for s in t.get("steps", []))
        keys[(t["file_index"], seq)] += 1
    n_dup_traces = sum(n for n in keys.values() if n > 1)
    print(f"\n--- 3. Near-duplicate traces (same procedure + identical action sequence) ---")
    print(f"  duplicates : {_pct(n_dup_traces, len(traces))} of all traces")


def main():
    traces = _load_traces()
    sft = _load_sft()
    print(f"Loaded {len(traces)} traces from {TRACES_PATH.name}")
    print(f"Loaded {len(sft)} SFT examples from {SFT_PATH.name}")

    step_level_balance(sft)
    corruption_distribution(traces)
    near_duplicates(traces)


if __name__ == "__main__":
    main()
