import json
import yaml
from pathlib import Path

here = Path(__file__).parent

with open(here / "held_out.json", encoding="utf-8") as f:
    records = json.load(f)

with open(here / "held_out.yaml", "w", encoding="utf-8") as f:
    yaml.dump(records, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"Converted {len(records)} records to held_out.yaml")
