#sanity-checks the PRM training data and prints a short report.
#runs five passes:
#  1. trace-level class balance (positive vs negative traces)
#  2. step-level class balance (Yes vs No after expansion)
#  3. per-procedure trace count distribution
#  4. corruption-type distribution
#  5. near-duplicate traces (same procedure + identical action sequence)
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


def trace_level_balance(traces: list[dict]) -> None:
    pos = sum(1 for t in traces if t.get("label") == 1)
    neg = sum(1 for t in traces if t.get("label") == 0)
    other = len(traces) - pos - neg
    print(f"\n--- 1. Trace-level class balance ({len(traces)} traces) ---")
    print(f"  positive (label=1) : {pos:5d}  {_pct(pos, len(traces))}  {_bar(pos, len(traces))}")
    print(f"  negative (label=0) : {neg:5d}  {_pct(neg, len(traces))}  {_bar(neg, len(traces))}")
    if other:
        print(f"  other / null label : {other:5d}  {_pct(other, len(traces))}")
    ratio = neg / pos if pos else 0
    print(f"  negative : positive ratio = 1 : {ratio:.2f}")


def step_level_balance(sft_records: list[dict]) -> None:
    yes = sum(1 for r in sft_records if r["messages"][-1]["content"].startswith("Yes"))
    no = sum(1 for r in sft_records if r["messages"][-1]["content"].startswith("No"))
    other = len(sft_records) - yes - no
    print(f"\n--- 2. Step-level class balance ({len(sft_records)} SFT examples) ---")
    print(f"  Yes : {yes:5d}  {_pct(yes, len(sft_records))}  {_bar(yes, len(sft_records))}")
    print(f"  No  : {no:5d}  {_pct(no, len(sft_records))}  {_bar(no, len(sft_records))}")
    if other:
        print(f"  ??? : {other:5d}  {_pct(other, len(sft_records))}")
    ratio = no / yes if yes else 0
    print(f"  No : Yes ratio = 1 : {ratio:.2f}")


def per_procedure_counts(traces: list[dict]) -> None:
    per_proc = Counter(t["file_index"] for t in traces)
    counts = list(per_proc.values())
    counts.sort()
    n_proc = len(per_proc)
    print(f"\n--- 3. Per-procedure trace counts ({n_proc} unique procedures) ---")
    print(f"  min/median/max traces per procedure : {min(counts)} / {counts[len(counts)//2]} / {max(counts)}")
    print(f"  mean : {sum(counts) / n_proc:.1f}")
    #show top 5 most over-represented procedures
    top = per_proc.most_common(5)
    print("  top 5 most-traced procedures:")
    for fidx, n in top:
        print(f"    file_index={fidx:>12}  {n} traces")
    #flag procedures with abnormally many traces (> 3× median)
    median = counts[len(counts) // 2]
    over = [fidx for fidx, n in per_proc.items() if n > 3 * median]
    if over:
        print(f"  WARN: {len(over)} procedures have >3× the median trace count")


def corruption_distribution(traces: list[dict]) -> None:
    types = Counter(t.get("corruption_type") for t in traces)
    print(f"\n--- 4. Corruption-type distribution ---")
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
    dup_groups = [(k, n) for k, n in keys.items() if n > 1]
    n_dup_traces = sum(n for _, n in dup_groups)
    print(f"\n--- 5. Near-duplicate traces (same procedure + identical action sequence) ---")
    print(f"  duplicate groups : {len(dup_groups)}")
    print(f"  traces involved  : {n_dup_traces} ({_pct(n_dup_traces, len(traces))} of all traces)")
    if dup_groups:
        worst = sorted(dup_groups, key=lambda x: -x[1])[:5]
        print("  worst 5 groups:")
        for (fidx, seq), n in worst:
            seq_preview = " ->".join(seq[:3]) + (" ->..." if len(seq) > 3 else "")
            print(f"    file_index={fidx}  ×{n}  [{seq_preview}]")


def main():
    traces = _load_traces()
    sft = _load_sft()
    print(f"Loaded {len(traces)} traces from {TRACES_PATH.name}")
    print(f"Loaded {len(sft)} SFT examples from {SFT_PATH.name}")

    trace_level_balance(traces)
    step_level_balance(sft)
    per_procedure_counts(traces)
    corruption_distribution(traces)
    near_duplicates(traces)


if __name__ == "__main__":
    main()
