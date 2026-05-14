#same passes as validate_training_data.py but pointed at the deduplicated/balanced files
#produced by prepare_prm_data_deduplicated.py. lets us sanity-check that the cleaning
#actually flattened the duplicate rate and improved class balance.
#
#  python validate_training_data_deduplicated.py


import json
from pathlib import Path

from validate_training_data import (
    step_level_balance,
    corruption_distribution,
    near_duplicates,
)


_HERE = Path(__file__).parent
TRACES_PATH = _HERE / "prm_training_data_deduplicated.json"
SFT_PATH    = _HERE / "prm_sft_train_deduplicated.jsonl"


def _load_traces() -> list[dict]:
    with open(TRACES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_sft() -> list[dict]:
    records = []
    with open(SFT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    traces = _load_traces()
    sft = _load_sft()
    print(f"Loaded {len(traces)} traces from {TRACES_PATH.name}")
    print(f"Loaded {len(sft)} SFT examples from {SFT_PATH.name}")

    step_level_balance(sft)
    corruption_distribution(traces)
    near_duplicates(traces)


if __name__ == "__main__":
    main()
