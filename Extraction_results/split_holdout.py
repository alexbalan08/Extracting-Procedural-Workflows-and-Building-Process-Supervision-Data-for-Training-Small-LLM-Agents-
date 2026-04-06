import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split clean predictions into PRM training set and held-out agent evaluation set")
    here = Path(__file__).parent
    parser.add_argument("--input", type=Path, default=here / "extraction_predictions.json")
    parser.add_argument("--holdout_output", type=Path, default=here / "held_out.json")
    parser.add_argument("--train_output", type=Path, default=here / "extraction_predictions.json")
    parser.add_argument("--holdout_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)

    #fix the random seed so the split is always reproducible
    #if we re-run this script we always get the same 50 held-out procedures
    #this is important so we dont accidentally leak held-out data later
    random.seed(args.seed)
    indices = list(range(total))
    random.shuffle(indices)

    holdout_indices = set(indices[: args.holdout_size])
    holdout = [records[i] for i in sorted(holdout_indices)]
    train = [records[i] for i in range(total) if i not in holdout_indices]

    #save held-out — these procedures are never used for PRM training
    #only used at the very end for agent evaluation
    with open(args.holdout_output, "w", encoding="utf-8") as f:
        json.dump(holdout, f, indent=2, ensure_ascii=False)

    #overwrite the clean predictions with only the 417 training procedures
    #from this point on extraction_predictions_clean.json is the prm training source
    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    print(f"Total:    {total}")
    print(f"Held-out: {len(holdout)}  → {args.holdout_output}")
    print(f"Training: {len(train)}  → {args.train_output}")
    print()
    print(f"Held-out file_indices:")
    for r in holdout:
        print(f"  {r['file_index']}")


if __name__ == "__main__":
    main()
