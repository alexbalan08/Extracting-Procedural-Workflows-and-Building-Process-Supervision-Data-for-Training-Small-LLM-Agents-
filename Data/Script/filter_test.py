

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM_Training" / "prompts"))
from system_prompt import EXTRACTION_SYSTEM_PROMPT
from feedback_prompt import format_initial_user_message

processed_dir = Path(__file__).parent.parent / "Processed"

MAX_INPUT_TOKENS = 1000  
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def count_input_tokens(tokenizer, procedure_text: str) -> int:
    messages = [
        {"role": "system",  "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",    "content": format_initial_user_message(procedure_text)},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    return tokenized["input_ids"].shape[-1]


def main():
    print(f"Loading tokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading extracted_test.json ...")
    with open(processed_dir / "extracted_test.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    kept, dropped = [], []
    for rec in records:
        n_tokens = count_input_tokens(tokenizer, rec["procedure_text"])
        if n_tokens <= MAX_INPUT_TOKENS:
            kept.append(rec)
        else:
            dropped.append((rec["file_index"], n_tokens))

    total = len(records)
    print(f"\nResults:")
    print(f"  Total   : {total}")
    print(f"  Kept    : {len(kept)} ({len(kept)/total*100:.1f}%)")
    print(f"  Dropped : {len(dropped)} ({len(dropped)/total*100:.1f}%)")
    if dropped:
        print(f"  Longest dropped: {max(t for _, t in dropped)} tokens")

    save = input("\nOverwrite extracted_test.json with filtered data? [y/N] ").strip().lower()
    if save == "y":
        with open(processed_dir / "extracted_test.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(kept)} records to extracted_test.json")
    else:
        print("Not saved.")


if __name__ == "__main__":
    main()
