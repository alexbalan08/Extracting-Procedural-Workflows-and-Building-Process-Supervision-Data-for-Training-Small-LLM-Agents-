import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter extraction predictions — removes records with no execution states or null workflow")
    _here = Path(__file__).parent
    parser.add_argument("--input", type=Path, default=_here / "extraction_predictions.json")
    parser.add_argument("--output", type=Path, default=_here / "extraction_predictions.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    removed = []
    kept = []

    #will filter out extractions with no execution states due to json parsin errors.
    #this will basically be garbage data and it represents only 17 out of 489 procedures extracted.
    #that is less than 3.5%
    #also filter procedures with over 100 execution states as these are loop explosions
    #from the path enumeration hitting MAX_LOOP_ITERATIONS not real data.
    #only 5 out of 489
    #so we get remaining 467 reaosning traces after extaraction

    for r in records:
        n = len(r.get("execution_states", []))
        if r.get("workflow") is None:
            removed.append((r["file_index"], "null workflow"))
        elif n == 0:
            removed.append((r["file_index"], "0 execution states"))
        elif n > 100:
            removed.append((r["file_index"], f"too complex ({n} execution states)"))
        else:
            kept.append(r)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"Total:   {total}")
    print(f"Removed: {len(removed)}")
    print(f"Kept:    {len(kept)}")
    print()
    if removed:
        print("Removed records:")
        for file_index, reason in removed:
            print(f"  file_index={file_index}  ({reason})")
    print()
    print(f"Saved {len(kept)} clean records to {args.output}")


if __name__ == "__main__":
    main()
