"""Filter extracted training records to only keep those with perfect validation scores
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Procedures_schema"))
from validate_extraction import validate_record

processed_dir = Path(__file__).parent.parent / "Processed"

METRIC_NAMES = [
    ("actions",              lambda m: m["actions"]["f1"]),
    ("gateway_type",         lambda m: m["gateways"]["type_accuracy"]),
    ("gateway_role",         lambda m: m["gateways"]["role_accuracy"]),
    ("action_successors",    lambda m: m["action_successors"]["f1"]),
    ("action_predecessors",  lambda m: m["action_predecessors"]["f1"]),
    ("gateway_next",         lambda m: m["gateway_branches_next"]["f1"]),
    ("gateway_incoming",     lambda m: m["gateway_incoming"]["f1"]),
    ("branch_tuples",        lambda m: m["branch_tuples"]["f1"]),
    ("branch_counts",        lambda m: m["branch_counts"]["accuracy"]),
    ("execution_states",     lambda m: m["execution_states"]["f1"]),
]

#we remove all entries under this threshold we can later experiment with 
THRESHOLD = 0.98

def is_perfect(metrics: dict) -> bool:
    return all(fn(metrics) >= THRESHOLD for _, fn in METRIC_NAMES)


def main():
    print("Loading merged_dataset.json ...")
    with open(processed_dir / "merged_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_by_index = {r["file_index"]: r for r in raw_data}

    print("Loading extracted_train.json ...")
    with open(processed_dir / "extracted_train.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    perfect, dropped = [], []
    failure_counts = {name: 0 for name, _ in METRIC_NAMES}

    for record in extracted:
        file_idx = record["file_index"]
        raw = raw_by_index.get(file_idx)
        if raw is None:
            dropped.append(file_idx)
            continue
        metrics = validate_record(raw, record)
        if is_perfect(metrics):
            perfect.append(record)
        else:
            dropped.append(file_idx)
            for name, fn in METRIC_NAMES:
                if fn(metrics) < THRESHOLD:
                    failure_counts[name] += 1

    total = len(extracted)
    num_dropped = len(dropped)
    print(f"\nResults:")
    print(f"  Total records   : {total}")
    print(f"  Perfect (kept)  : {len(perfect)} ({len(perfect)/total*100:.1f}%)")
    print(f"  Dropped         : {num_dropped} ({num_dropped/total*100:.1f}%)")

    print(f"\nFailure breakdown (which metric caused the drop):")
    for name, _ in METRIC_NAMES:
        count = failure_counts[name]
        if count > 0:
            print(f"  {name:<25s}: {count} records ({count/num_dropped*100:.1f}% of dropped)")

    save = input("\nSave filtered data? [y/N] ").strip().lower()
    if save == "y":
        output_path = processed_dir / "extracted_train.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(perfect, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")
    else:
        print("Not saved.")


if __name__ == "__main__":
    main()
