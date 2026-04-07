import json
import random
import statistics
from pathlib import Path

# I will use this class for splitting both the initial sequenceflow and also my extraction

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Processed"

INPUT_PATH = PROCESSED_DIR / "extracted_workflows.json"
TRAIN_RATIO = 0.80
SEED = 42

# Filter procedures where the ground truth workflow JSON exceeds a token limit of let s say 2500 tokens
# This avoids model running out of output tokens during extraction and keeps the costs ok for me
# the median of tokens length from ground truth is 1040 tokens so i will cap a limit of 2500 to still maintain some space for longer procedures
# around 800 tokens-1000 will be added from the cot generation so we will be around 3500 tokens max per procedure to output
# under the limit of 4096 tokens i set
MAX_JSON_TOKENS = 2500  # to test with


def count_tokens(workflow: dict) -> int:
    # chars//4 = tokens approx.
    return len(json.dumps(workflow, ensure_ascii=False)) // 4


def split(data):
    random.seed(SEED)
    random.shuffle(data)
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    print(f"Total: {len(data)} | Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data


def filter_split(records, name, max_tokens=MAX_JSON_TOKENS):
    kept, dropped = [], []
    for record in records:
        tokens = count_tokens(record["workflow"])
        if tokens <= max_tokens:
            kept.append(record)
        else:
            dropped.append((record["file_index"], tokens))

    total = len(records)
    print(f"\n{name} — MAX_JSON_TOKENS={max_tokens}")
    print(f"  Total   : {total}")
    print(f"  Kept    : {len(kept)}  ({len(kept)/total*100:.1f}%)")
    print(f"  Dropped : {len(dropped)}  ({len(dropped)/total*100:.1f}%)")

    if kept:
        sizes = [count_tokens(r["workflow"]) for r in kept]
        print(f"  Kept median tokens: {int(statistics.median(sizes))}")
        print(f"  Kept max tokens   : {max(sizes)}")

    return kept


if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data, test_data = split(data)

    for name, subset in [("train", train_data), ("test", test_data)]:
        filtered = filter_split(subset, name)
        output_path = PROCESSED_DIR / f"extracted_{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {output_path}")
