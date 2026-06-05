#same passes as validate_training_data.py but pointed at the deduplicated/balanced files
#produced by prepare_prm_data_deduplicated.py. lets us sanity-check that the cleaning
#actually flattened the duplicate rate and improved class balance.
#
#  python validate_training_data_deduplicated.py


from pathlib import Path

from validate_training_data import (
    _load_traces,
    _load_sft,
    step_level_balance,
    corruption_distribution,
    near_duplicates,
)


_HERE = Path(__file__).parent
TRACES_PATH = _HERE / "prm_training_data_deduplicated.json"
SFT_PATH    = _HERE / "prm_sft_train_deduplicated.jsonl"


def main():
    traces = _load_traces(TRACES_PATH)
    sft = _load_sft(SFT_PATH)
    print(f"Loaded {len(traces)} traces from {TRACES_PATH.name}")
    print(f"Loaded {len(sft)} SFT examples from {SFT_PATH.name}")

    step_level_balance(sft)
    corruption_distribution(traces)
    near_duplicates(traces)


if __name__ == "__main__":
    main()
