import json
from pathlib import Path

raw_dir = Path(__file__).parent.parent / 'Raw'
output_dir = Path(__file__).parent.parent / 'Processed'
output_dir.mkdir(exist_ok=True)


def clean_text(text):
    """Clean procedure text to produce raw text, as we would expect to find in manuals.
    Removes <SEP> tokens (separator between chunks/actors in PAGGED) and add space instread
    Replaces newlines with spaces
    Collapses multiple spaces into one
    """
    if not isinstance(text, str):
        return text
    import re
    text = text.replace(' <SEP> ', ' ').replace('<SEP>', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(r' {2,}', ' ', text)
    return text


def clean_dataset(data):
    """Clean all text fields in the dataset."""
    if isinstance(data, list):
        return [clean_dataset(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_text(value) if key == 'paragraph' else clean_dataset(value)
                for key, value in data.items()}
    return data


with open(raw_dir / 'dev.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

with open(raw_dir / 'test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open(raw_dir / 'train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)


dev_data = clean_dataset(dev_data)
test_data = clean_dataset(test_data)
train_data = clean_dataset(train_data)


if isinstance(dev_data, list) and isinstance(test_data, list) and isinstance(train_data, list):
    merged_data = dev_data + test_data + train_data
    #we have the correct number of records 3394 so all good
    print(f"Merged {len(dev_data)} + {len(test_data)} + {len(train_data)} = {len(merged_data)} records")
else:
    merged_data = {
        'dev': dev_data,
        'test': test_data,
        'train': train_data
    }

# Remove procedures with more than 500 words
before = len(merged_data)
merged_data = [r for r in merged_data if len(r['paragraph'].split()) <= 500]
print(f"Removed {before - len(merged_data)} procedures with >500 words ({len(merged_data)} remaining)")

output_path = output_dir / 'merged_dataset.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2)

print(f"saved {output_path}")
