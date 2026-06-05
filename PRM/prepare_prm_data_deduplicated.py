#same pipeline as prepare_prm_data.py but with deduplication for mixed labels, 
#all the same steps of histoy with same actions possible will colllapse to one 

#we also subsample positives at the SFT level until Yes count == No count so we have fully balanced data
#output is the same SFT format 



import json
import random
from pathlib import Path

from prepare_prm_data import expand_to_sft

#matches the seed used in build_negatives.py and also for subsampling
_SEED = 42

#target SFT size after balance — 4200 per class = 8400 total, 43% smaller than the original
#prm_sft_train.jsonl (14651)
TARGET_PER_CLASS = 4200


_here = Path(__file__).parent
PRM_DATA_PATH    = _here / "prm_training_data.json"
PREDICTIONS_PATH = _here.parent / "Extraction_results" / "extraction_predictions.json"
TRACES_OUT       = _here / "prm_training_data_deduplicated.json"
SFT_OUT          = _here / "prm_sft_train_deduplicated.jsonl"


#if we have the same history and available actions but diff labels we keep the positive since we know alreasdy 
#it s correct.
def deduplicate(traces: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for t in traces:
        seq = tuple(s["action"] for s in t.get("steps", []))
        groups.setdefault((t["file_index"], seq), []).append(t)

    cleaned = []
    for ts in groups.values():
        pos = [t for t in ts if t.get("label") == 1]
        neg = [t for t in ts if t.get("label") == 0]
        cleaned.append(pos[0] if pos else neg[0])
    return cleaned


def main():
    with open(PRM_DATA_PATH, encoding="utf-8") as f:
        traces = json.load(f)

    n_before = len(traces)
    traces = deduplicate(traces)
    n_after = len(traces)
    pct_dropped = 100.0 * (n_before - n_after) / max(n_before, 1)
    print(f"Traces (dedup): {n_before} -> {n_after}  ({pct_dropped:.1f}% dropped)")

    
    with open(TRACES_OUT, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    #SFT expansion is identical to prepare_prm_data.py 
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        predictions = json.load(f)

    examples, kept, dropped, contradictions = expand_to_sft(traces, predictions)

    
    #shuffle the final mix so positives and negatives are interleaved 
    yes = [e for e in examples if e["messages"][-1]["content"] == "Yes"]
    no  = [e for e in examples if e["messages"][-1]["content"].startswith("No")]
    n_yes_before, n_no_before = len(yes), len(no)

    rng = random.Random(_SEED)
    if len(yes) > len(no):
        yes = rng.sample(yes, len(no))
    elif len(no) > len(yes):
        no  = rng.sample(no, len(yes))
    

    #to make it fully balancded
    if TARGET_PER_CLASS is not None and len(yes) > TARGET_PER_CLASS:
        yes = rng.sample(yes, TARGET_PER_CLASS)
        no  = rng.sample(no,  TARGET_PER_CLASS)
    balanced = yes + no
    rng.shuffle(balanced)

    with open(SFT_OUT, "w", encoding="utf-8") as f:
        for ex in balanced:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Step examples    : kept={kept}  dropped={dropped}  contradictions_filtered={contradictions}")
    print(f"  Yes (before balance) : {n_yes_before}")
    print(f"  No  (before balance) : {n_no_before}")
    print(f"  Yes (after balance)  : {len(yes)}")
    print(f"  No  (after balance)  : {len(no)}")
    print(f"  Ratio                : 1 : {len(no) / max(len(yes), 1):.2f}")
    print(f"Saved traces to {TRACES_OUT}")
    print(f"Saved SFT to    {SFT_OUT}")


if __name__ == "__main__":
    main()
