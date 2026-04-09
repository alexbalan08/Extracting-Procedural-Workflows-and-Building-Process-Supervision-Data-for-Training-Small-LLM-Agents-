#converts extracted_test.json ground truth to yaml so we can validate yaml extraction against yaml ground truth
#one shot script, run it once before running the yaml extraction pipeline

import json
import yaml
from pathlib import Path


def main():
    here = Path(__file__).parent
    input_path = here.parent / "Data" / "Processed" / "extracted_test.json"
    output_path = here.parent / "Data" / "Processed" / "extracted_test.yaml"

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"Converted {len(records)} records")
    print(f"  {input_path} → {output_path}")


if __name__ == "__main__":
    main()
