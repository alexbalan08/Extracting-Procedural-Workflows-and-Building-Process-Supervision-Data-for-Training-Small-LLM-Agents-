"""
Filter procedures where the ground truth workflow JSON exceeds a token limit.
This avoids GPT-4o running out of output tokens during extraction.

Token budget reasoning:
  - Ground truth JSON tokens  ~= len(workflow_json) / 4
  - CoT reasoning estimate    ~= 870 tokens (avg ~8.8 actions * ~400 chars/action / 4)
  - Total output estimate     ~= JSON tokens + 870
  We filter at MAX_JSON_TOKENS so total stays safely under the model's max_new_tokens.
"""

import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"

MAX_JSON_TOKENS = 2000  # filter out records whose ground truth JSON exceeds this


def count_tokens(workflow: dict) -> int:
    """Rough token estimate: chars / 4."""
    return len(json.dumps(workflow, ensure_ascii=False)) // 4


def filter_file(input_path: Path, output_path: Path, max_tokens: int = MAX_JSON_TOKENS):
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    kept, dropped = [], []
    for record in records:
        tokens = count_tokens(record["workflow"])
        if tokens <= max_tokens:
            kept.append(record)
        else:
            dropped.append((record["file_index"], tokens))

    total = len(records)
    print(f"\n{input_path.name} — MAX_JSON_TOKENS={max_tokens}")
    print(f"  Total   : {total}")
    print(f"  Kept    : {len(kept)}  ({len(kept)/total*100:.1f}%)")
    print(f"  Dropped : {len(dropped)}  ({len(dropped)/total*100:.1f}%)")

    if kept:
        sizes = [count_tokens(r["workflow"]) for r in kept]
        print(f"  Kept median tokens: {int(statistics.median(sizes))}")
        print(f"  Kept max tokens   : {max(sizes)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    filter_file(
        input_path=PROCESSED_DIR / "extracted_test.json",
        output_path=PROCESSED_DIR / "extracted_test_filtered.json",
    )
    filter_file(
        input_path=PROCESSED_DIR / "extracted_train.json",
        output_path=PROCESSED_DIR / "extracted_train_filtered.json",
    )
