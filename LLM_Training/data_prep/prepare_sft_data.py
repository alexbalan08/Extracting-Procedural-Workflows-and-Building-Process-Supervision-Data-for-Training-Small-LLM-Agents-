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


"""In this class I will convert the extracted workflows (gold) into supervised records for training a small LLM to do the extraction itself from only raw 
text and produce us json structure similar to what we expect."""

def _describe_ref(ref: str, action_map: dict) -> str:
    if ref == "start":
        return "the process start"
    if ref in action_map:
        return f"'{action_map[ref]['name']}'"
    return f"gateway '{ref}'"


def _gateway_type_explanation(gw_type: str) -> str:
    #to help the CoT genegration
    if gw_type == "exclusive":
        return "only one branch is taken (exclusive/XOR)"
    elif gw_type == "parallel":
        return "all branches execute simultaneously (parallel/AND)"
    elif gw_type == "inclusive":
        return "one or more branches may be taken (inclusive/OR)"
    return gw_type


#Self-Refine" (Madaan et al., 2023)


 #i will force CoT for the model to learn how to reason
 #this will teach the model how to think about the workflow structure and how to describe it in a reasoning trace.
 #it will be used as part of the response during training so the model learns to produce this kind of reasoning when given a procedure text
def generate_reasoning_trace(workflow: dict) -> str:

    #in here we store for the model how to identify components and use them to reason step by step
    actions = workflow.get("actions", [])
    gateways = workflow.get("gateways", [])

    exec_states = workflow.get("execution_states", [])

    action_map = {a["id"]: a for a in actions}
    gw_set = {g["id"] for g in gateways}
    lines = ["Let me analyze this procedure step by step.", ""]


    #actions - show how IDs are derived so the model learns the naming convention
    lines.append("**Step 1 — Extract actions and assign IDs**")
    start_actions = [a for a in actions if "start" in a.get("predecessors", [])]
    end_actions = [a for a in actions if not a.get("successors", [])]
    lines.append(f"Reading through the text, I find {len(actions)} distinct action(s):")
    for action in actions:
        actor_note = f", performed by {action['actor']}" if action.get("actor") else ""
        lines.append(f"  - '{action['name']}' → id: {action['id']}{actor_note}")
    lines.append("")

    
    #build the flow step by step so the model learns to connect actions
    lines.append("**Step 2 — Determine the flow between actions**")
    if start_actions:
        names = ", ".join(f"'{a['name']}'" for a in start_actions)
        lines.append(f"The process begins with: {names}.")

    for action in actions:
        preds = action.get("predecessors", [])
        succs = action.get("successors", [])
        pred_strs = [_describe_ref(p, action_map) for p in preds]
        succ_parts = []
        for s in succs:
            if s in action_map:
                succ_parts.append(f"'{action_map[s]['name']}'")
            elif s in gw_set:
                succ_parts.append(f"a decision/split point ({s})")
            else:
                succ_parts.append(s)
        pred_str = ", ".join(pred_strs) if pred_strs else "nothing"
        succ_str = ", ".join(succ_parts) if succ_parts else "the process end"
        lines.append(f"  '{action['name']}': after {pred_str} → then {succ_str}")

    if end_actions:
        names = ", ".join(f"'{a['name']}'" for a in end_actions)
        lines.append(f"The process ends after: {names}.")
    lines.append("")

    
    #gateways - explain WHY a type is chosen and trace each branch
    lines.append("**Step 3 — Identify decision points and parallel splits**")
    if gateways:
        for gw in gateways:
            branches = gw.get("branches", [])
            incoming = gw.get("incoming_from", [])
            inc_strs = [_describe_ref(i, action_map) for i in incoming]
            lines.append(
                f"Gateway {gw['id']}: {_gateway_type_explanation(gw['type'])}."
            )
            lines.append(f"  Role: {gw['role']} (incoming from {', '.join(inc_strs)})")
            for i, branch in enumerate(branches, 1):
                cond = branch.get("condition", "default")
                next_ref = branch.get("next")
                if next_ref is None:
                    target = "process end"
                elif next_ref in action_map:
                    target = f"'{action_map[next_ref]['name']}'"
                else:
                    target = next_ref
                lines.append(f"  Branch {i}: [{cond}] → {target}")
    else:
        lines.append("No decision points or parallel splits — this is a straight linear sequence.")
    lines.append("")

    
    #for ex states show terminal paths so the model learns how to derive them
    lines.append("**Step 4 — Derive execution states**")
    terminal_states = [s for s in exec_states if s.get("can_terminate")]
    non_terminal = len(exec_states) - len(terminal_states)
    lines.append(
        f"Tracing all possible paths through the workflow produces "
        f"{len(exec_states)} reachable state(s): "
        f"{non_terminal} intermediate and {len(terminal_states)} terminal."
    )
    if terminal_states:
        for ts in terminal_states[:3]:
            completed = ts.get("completed_actions", [])
            if completed:
                lines.append(
                    f"  Terminal state after: {' → '.join(completed)}"
                )
        if len(terminal_states) > 3:
            lines.append(f"  ... and {len(terminal_states) - 3} more terminal state(s).")

    return "\n".join(lines)


#this simply build for each record from training dataset a json with messages field containing the system prompt
#user prompt (with procedure text) and model response (with reasoning trace and workflow json) that will be used
# for training the model to do the extraction itself

def build_sft_record(record: dict) -> dict:
    procedure_text = record["procedure_text"]
    workflow = record["workflow"]

    reasoning = generate_reasoning_trace(workflow)
    workflow_json = json.dumps(workflow, indent=2, ensure_ascii=False)

    assistant_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
        f"```json\n{workflow_json}\n```"
    )
    

    #here 
    return {
        "file_index": record["file_index"],
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}, #check the prompts folder
            {"role": "user", "content": format_initial_user_message(procedure_text)},
            {"role": "assistant", "content": assistant_content},
        ],
    }



def main():
    parser = argparse.ArgumentParser(description="Prepare supervised dataset from extracted workflows.")
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

    print(f"Wrote {len(data)} supervised records to {args.output}")


if __name__ == "__main__":
    main()
