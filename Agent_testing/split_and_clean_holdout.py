import argparse
import json
import random
from pathlib import Path


def main():
    here = Path(__file__).parent
    extraction_results = here.parent / "Extraction_results"

    parser = argparse.ArgumentParser(description="Split clean predictions into PRM training set and held-out agent evaluation set")
    parser.add_argument("--input", type=Path, default=extraction_results / "extraction_predictions.json")
    parser.add_argument("--holdout_output", type=Path, default=here / "held_out.json")
    parser.add_argument("--train_output", type=Path, default=extraction_results / "extraction_predictions.json")
    parser.add_argument("--holdout_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)

    #without this we d pick a different 50 every time the input size changes and leak data into PRM training
    holdout_file_indices = None
    if args.holdout_output.exists():
        with open(args.holdout_output, encoding="utf-8") as f:
            existing = json.load(f)
        holdout_file_indices = {r["file_index"] for r in existing}
        print(f"Found existing {args.holdout_output.name} with {len(holdout_file_indices)} file_indices — reusing")

    if holdout_file_indices is None:
        #random seed so the split is always reproducible
        #if we rerun this script from scratch we always get the same 50 held-out procedures
        random.seed(args.seed)
        indices = list(range(total))
        random.shuffle(indices)
        holdout_idx_set = set(indices[: args.holdout_size])
        holdout_records = [records[i] for i in sorted(holdout_idx_set)]
        holdout_file_indices = {r["file_index"] for r in holdout_records}
    else:
        holdout_records = [r for r in records if r["file_index"] in holdout_file_indices]

    train = [r for r in records if r["file_index"] not in holdout_file_indices]

    if len(train) == total and not holdout_records:
        print(f"Held-out indices already removed from {args.input.name} — nothing to do")
        return


    #depending on what version we use of the testin agent we want to give them acces 
    #only to the procedure text or also to the list of actions extracted 
    #or even to the execution graph (for the ensemble agent that consults it as a tool)
    cleaned = []
    for r in holdout_records:
        wf = r.get("workflow") or {}
        actions = [a["name"] for a in wf.get("actions", [])]
        conditions = sorted({
            b["condition"]
            for gw in wf.get("gateways", [])
            for b in gw.get("branches", [])
            if b.get("condition")
        })
        cleaned.append({
            "file_index":     r["file_index"],
            "procedure_text": r["procedure_text"],
            "actions":        actions,
            "conditions":     conditions,
        })

    #these procedures are never used for PRM training
    #only used at the very end for agent evaluation
    with open(args.holdout_output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    print(f"Total:    {total}")
    print(f"Held-out: {len(cleaned)}  → {args.holdout_output}")
    print(f"Training: {len(train)}  → {args.train_output}")
    print()
    print("Held-out file_indices:")
    for r in cleaned:
        print(f"  {r['file_index']}")


if __name__ == "__main__":
    main()
