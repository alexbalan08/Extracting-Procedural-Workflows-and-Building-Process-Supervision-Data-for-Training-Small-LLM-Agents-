"""Filter extracted training records by token budget.
Keeps only records whose SFT entry (system + user + assistant) fits within MAX_TOKENS.
Token count is estimated as total_chars // 4 (standard approximation for English+JSON).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM_Training" / "prompts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM_Training" / "data_prep"))
from prepare_sft_data import build_sft_record

processed_dir = Path(__file__).parent.parent / "Processed"


MAX_TOKENS = 4700


def estimate_tokens(sft_record: dict) -> int:
    total_chars = sum(len(m["content"]) for m in sft_record["messages"])
    return total_chars // 4


def main():
    print("Loading extracted_train.json ...")
    with open(processed_dir / "extracted_train.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    kept, dropped = [], []

    for record in extracted:
        sft = build_sft_record(record)
        tokens = estimate_tokens(sft)
        if tokens <= MAX_TOKENS:
            kept.append(record)
        else:
            dropped.append((record["file_index"], tokens))

    total = len(extracted)
    print(f"\nResults (MAX_TOKENS={MAX_TOKENS}):")
    print(f"  Total records : {total}")
    print(f"  Kept          : {len(kept)} ({len(kept)/total*100:.1f}%)")
    print(f"  Dropped       : {len(dropped)} ({len(dropped)/total*100:.1f}%)")

    if dropped:
        dropped.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Largest dropped records (file_index, tokens):")
        for fid, t in dropped[:5]:
            print(f"    {fid}: ~{t} tokens")

    output_path = processed_dir / "extracted_train.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(kept)} records to {output_path}")


if __name__ == "__main__":
    main()
