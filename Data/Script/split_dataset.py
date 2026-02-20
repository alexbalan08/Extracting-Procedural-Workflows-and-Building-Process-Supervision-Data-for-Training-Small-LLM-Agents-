import json
import random
from pathlib import Path

processed_dir = Path(__file__).parent.parent / 'Processed'
input_path = processed_dir / 'merged_dataset.json'

TRAIN_RATIO = 0.8
SEED = 42

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(SEED)
random.shuffle(data)

split_idx = int(len(data) * TRAIN_RATIO)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Total: {len(data)} | Train: {len(train_data)} | Test: {len(test_data)}")

for name, subset in [('train', train_data), ('test', test_data)]:
    output_path = processed_dir / f'{name}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_path}")
