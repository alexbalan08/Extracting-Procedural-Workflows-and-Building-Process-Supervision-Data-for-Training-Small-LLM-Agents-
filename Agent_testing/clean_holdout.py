import json
from pathlib import Path

here = Path(__file__).parent

#load the held-out procedures
with open(here / "held_out.json", encoding="utf-8") as f:
    records = json.load(f)

#remove extraction metadata that is not needed for agent testing
#we only keep what the agent and evaluator actually need:
#  file_index — identifier
#  procedure_text — input to the agent
#  workflow — the structured procedure (useful for debugging and analysis)
#  execution_states — ground truth for evaluating agent steps
KEEP = {"file_index", "procedure_text", "workflow", "execution_states"}

cleaned = [
    {k: v for k, v in r.items() if k in KEEP}
    for r in records
]

with open(here / "held_out.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"Cleaned {len(cleaned)} records — removed: reasoning, attempt, remaining_issues")
print(f"Saved to held_out.json")
