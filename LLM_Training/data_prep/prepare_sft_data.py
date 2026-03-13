"""
Convert extracted_train.json into a supervised fine-tuning (SFT) dataset.

Each record becomes a chat with:
  system  : EXTRACTION_SYSTEM_PROMPT
  user    : procedure_text
  assistant: <reasoning>...</reasoning>\n```json\n{workflow}\n```
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))
from system_prompt import EXTRACTION_SYSTEM_PROMPT
from feedback_prompt import format_initial_user_message

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Data" / "Processed" / "extracted_train.json"
DEFAULT_OUTPUT = Path(__file__).parent / "sft_train.jsonl"


# ── Reasoning trace generator ─────────────────────────────────────────────────

def _describe_ref(ref: str, action_map: dict) -> str:
    if ref == "start":
        return "the process start"
    if ref in action_map:
        return f"'{action_map[ref]['name']}'"
    return f"gateway '{ref}'"


def generate_reasoning_trace(workflow: dict) -> str:
    actors = workflow.get("actors", [])
    actions = workflow.get("actions", [])
    gateways = workflow.get("gateways", [])
    exec_states = workflow.get("execution_states", [])

    action_map = {a["id"]: a for a in actions}
    lines = ["Let me analyze this procedure step by step.", ""]

    # Actors
    if actors:
        lines.append(f"**Actors**: I identify {len(actors)} actor(s): {', '.join(actors)}.")
    else:
        lines.append("**Actors**: No specific actors are mentioned in this procedure.")
    lines.append("")

    # Actions
    start_actions = [a for a in actions if "start" in a.get("predecessors", [])]
    end_actions = [a for a in actions if not a.get("successors", [])]
    lines.append(f"**Actions**: I identify {len(actions)} action(s) in total.")
    if start_actions:
        names = ", ".join(f"'{a['name']}'" for a in start_actions)
        lines.append(f"The process starts with: {names}.")
    if end_actions:
        names = ", ".join(f"'{a['name']}'" for a in end_actions)
        lines.append(f"The process ends after: {names}.")
    lines.append("")

    lines.append("Action-by-action flow:")
    for action in actions:
        pred_strs = [_describe_ref(p, action_map) for p in action.get("predecessors", [])]
        succ_strs = [_describe_ref(s, action_map) for s in action.get("successors", [])]
        actor_note = f" (performed by {action['actor']})" if action.get("actor") else ""
        pred_str = ", ".join(pred_strs) if pred_strs else "nothing"
        succ_str = ", ".join(succ_strs) if succ_strs else "the process end"
        lines.append(
            f"  - '{action['name']}' [id={action['id']}]{actor_note}: "
            f"follows {pred_str} → leads to {succ_str}."
        )
    lines.append("")

    # Gateways
    if gateways:
        lines.append(f"**Gateways**: I identify {len(gateways)} gateway(s):")
        for gw in gateways:
            branches = gw.get("branches", [])
            conds = [b.get("condition", "default") for b in branches]
            lines.append(
                f"  - {gw['id']}: {gw['type']} gateway ({gw['role']}) with "
                f"{len(branches)} branch(es). Conditions: {conds}."
            )
    else:
        lines.append("**Gateways**: No decision or parallel split points — this is a linear sequence.")
    lines.append("")

    # Execution states summary
    terminal_count = sum(1 for s in exec_states if s.get("can_terminate"))
    lines.append(
        f"**Execution states**: The workflow produces {len(exec_states)} execution state(s), "
        f"of which {terminal_count} is/are terminal (process can end)."
    )

    return "\n".join(lines)


# ── SFT record builder ────────────────────────────────────────────────────────

def build_sft_record(record: dict) -> dict:
    procedure_text = record["procedure_text"]
    workflow = record["workflow"]

    reasoning = generate_reasoning_trace(workflow)
    workflow_json = json.dumps(workflow, indent=2, ensure_ascii=False)

    assistant_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
        f"```json\n{workflow_json}\n```"
    )

    return {
        "file_index": record["file_index"],
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": format_initial_user_message(procedure_text)},
            {"role": "assistant", "content": assistant_content},
        ],
    }



def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset from extracted workflows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in data:
            sft = build_sft_record(record)
            f.write(json.dumps(sft, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} SFT records to {args.output}")


if __name__ == "__main__":
    main()
