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
    return f"gateway {ref}"


#Self-Refine" (Madaan et al., 2023)


 #i will force CoT for the model to learn how to reason
 #this will teach the model how to think about the workflow structure and how to describe it in a reasoning trace.
 #it will be used as part of the response during training so the model learns to produce this kind of reasoning when given a procedure text
 #NOTE: ID conventions, schema fields, and gateway patterns are already in the system prompt
 #so here we only show instance-specific reasoning (what was found and why)
def generate_reasoning_trace(workflow: dict) -> str:

    actions = workflow.get("actions", [])
    gateways = workflow.get("gateways", [])
    exec_states = workflow.get("execution_states", [])

    action_map = {a["id"]: a for a in actions}
    lines = ["Let me analyze this procedure step by step.", ""]

    #step 1: list what actions were found and their IDs
    lines.append("**Step 1 — Extract actions**")
    lines.append(f"I find {len(actions)} distinct action(s) in the text:")
    for i, action in enumerate(actions, 1):
        actor_note = f", performed by {action['actor']}" if action.get("actor") else ""
        lines.append(f"  {i}. \"{action['name']}\" → {action['id']}{actor_note}")
    lines.append("")

    #step 2: show how actions connect — the instance-specific flow reasoning
    lines.append("**Step 2 — Determine the flow**")
    start_actions = [a for a in actions if "start" in a.get("predecessors", [])]
    end_actions = [a for a in actions if not a.get("successors", [])]

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
            else:
                succ_parts.append(f"gateway {s}")
        pred_str = ", ".join(pred_strs) if pred_strs else "nothing"
        succ_str = ", ".join(succ_parts) if succ_parts else "end of process"
        lines.append(f"  {action['id']}: after {pred_str} → then {succ_str}")

    if end_actions:
        names = ", ".join(f"'{a['name']}'" for a in end_actions)
        lines.append(f"The process ends after: {names}.")
    lines.append("")

    #step 3: gateways — explain WHY a type was chosen from the text
    lines.append("**Step 3 — Identify gateways**")
    if gateways:
        lines.append(f"I find {len(gateways)} gateway(s):")
        for gw in gateways:
            branches = gw.get("branches", [])
            incoming = gw.get("incoming_from", [])
            inc_strs = [_describe_ref(i, action_map) for i in incoming]

            lines.append(f"  {gw['id']} ({gw['type']}, {gw['role']}):")
            lines.append(f"    Incoming from: {', '.join(inc_strs)}")
            for i, branch in enumerate(branches, 1):
                cond = branch.get("condition")
                next_ref = branch.get("next")
                if next_ref is None:
                    target = "process ends"
                elif next_ref in action_map:
                    target = f"'{action_map[next_ref]['name']}'"
                else:
                    target = f"gateway {next_ref}"
                cond_str = f" when \"{cond}\"" if cond else ""
                lines.append(f"    Branch {i}: → {target}{cond_str}")
    else:
        lines.append("No gateways — straight linear sequence.")
    lines.append("")

    #step 4: trace execution paths to show how states are derived
    lines.append("**Step 4 — Derive execution states**")
    terminal_states = [s for s in exec_states if s.get("can_terminate")]

    #trace one full path step by step so the model learns the derivation
    if terminal_states:
        example = terminal_states[0]
        completed = example.get("completed_actions", [])
        conditions = example.get("conditions_met", [])

        lines.append("Tracing one path through the workflow:")
        for step_idx in range(len(completed)):
            done_so_far = completed[:step_idx]
            next_action = completed[step_idx]
            done_str = ", ".join(done_so_far) if done_so_far else "(none)"
            lines.append(f"  After [{done_str}] → next: {next_action}")
        #terminal
        cond_note = f" (conditions: {', '.join(conditions)})" if conditions else ""
        lines.append(f"  After [{', '.join(completed)}] → process can terminate{cond_note}")

    lines.append("")
    non_terminal = len(exec_states) - len(terminal_states)
    lines.append(f"Across all paths: {len(exec_states)} states total "
                 f"({non_terminal} intermediate, {len(terminal_states)} terminal).")

    if len(terminal_states) > 1:
        lines.append("Terminal paths:")
        for ts in terminal_states[:5]:
            completed = ts.get("completed_actions", [])
            conds = ts.get("conditions_met", [])
            cond_note = f" (when: {', '.join(conds)})" if conds else ""
            lines.append(f"  [{' → '.join(completed)}]{cond_note}")
        if len(terminal_states) > 5:
            lines.append(f"  ... and {len(terminal_states) - 5} more.")

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
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
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
