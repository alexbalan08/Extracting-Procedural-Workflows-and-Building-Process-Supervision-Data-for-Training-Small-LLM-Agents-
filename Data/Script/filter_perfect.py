"""
we keep only records where SFT entry (system + user + assistant) fits within MAX_TOKENS
Token count uses the real Llama tokenizer for accuracy.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM_Training" / "prompts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM_Training" / "data_prep"))
from prepare_sft_data import build_sft_record

processed_dir = Path(__file__).parent.parent / "Processed"

MAX_TOKENS = 6144
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def count_tokens(sft_record: dict, tokenizer) -> int:
    text = tokenizer.apply_chat_template(sft_record["messages"], tokenize=False)
    return len(tokenizer.encode(text))


def main():
    print(f"Loading tokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading extracted_train.json ...")
    with open(processed_dir / "extracted_train.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    kept, dropped = [], []

    for record in extracted:
        sft = build_sft_record(record)
        tokens = count_tokens(sft, tokenizer)
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
    

    #i will overwrite the same file so then i generate the sft traces
    output_path = processed_dir / "extracted_train.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(kept)} records to {output_path}")


if __name__ == "__main__":
    main()
