
#Converts prm_training_data.json into step-level SFT records for PRM training.
#
#Each trace is expanded into one training example per step:
#  - steps_so_far: actions correctly completed before this step (empty at the start)
#  - candidate_next: the action being scored at this step
#  - available_actions: all actions extracted from the workflow (always visible)
#  - label: "Yes" (label=1) or "No" (label=0)
#
#At inference the agent starts with steps_so_far=[] and scores every remaining action
#as a candidate, picking the one with highest P("Yes"). This repeats until the
#procedure is complete.
#
#Token budget: we estimate tokens as total_chars // 4 and drop examples over MAX_TOKENS
#to stay within the 5120 context window used during training.

import json
from pathlib import Path

_here = Path(__file__).parent
PRM_DATA_PATH    = _here / "prm_training_data.json"
PREDICTIONS_PATH = _here.parent / "Extraction_results" / "extraction_predictions.json"
OUTPUT_PATH      = _here / "prm_sft_train.jsonl"

MAX_TOKENS = 4700  # 5120 max_seq_length minus chat template overhead and approximation buffer

PRM_SYSTEM_PROMPT = (
    "You are a process reward model for procedural workflows. "
    "Given a procedure description, the full list of available actions, "
    "and the steps completed so far, decide whether the proposed next action "
    "is correct at this point in the procedure. "
    "Answer only \"Yes\" or \"No\"."
)


def estimate_tokens(record: dict) -> int:
    #chars//4 = tokens approx.
    total_chars = sum(len(m["content"]) for m in record["messages"])
    return total_chars // 4


def generate_answer(label: int, step_index: int, corruption_type, corruption_detail) -> str:
    if label == 1:
        return "Yes"

    #at the first wrong step we add a brief deterministic explanation of the error type
    #this is zero-cost (metadata already exists) and 100% accurate (not LLM-generated)
    #at inference we only read the first token logit (Yes/No) so the explanation
    #does not affect scoring — it only helps the model learn error patterns during training
    if corruption_detail and step_index == corruption_detail.get("first_wrong_step"):
        if corruption_type == "skip_action":
            return f"No. Error: skip_action. Action '{corruption_detail['skipped_action']}' was skipped."
        elif corruption_type == "swap_adjacent":
            return (f"No. Error: swap_adjacent. "
                    f"'{corruption_detail['swapped'][0]}' and '{corruption_detail['swapped'][1]}' were swapped.")
        elif corruption_type == "wrong_branch":
            return (f"No. Error: wrong_branch. "
                    f"Took '{corruption_detail['wrong_next']}' instead of '{corruption_detail['correct_next']}'.")
        elif corruption_type == "premature_stop":
            return (f"No. Error: premature_stop. "
                    f"Execution ended at step {corruption_detail['stopped_at']} of {corruption_detail['full_length']}.")
        elif corruption_type == "repeat_completed":
            return (f"No. Error: repeat_completed. "
                    f"Action '{corruption_detail['repeated_action']}' was already completed earlier.")
        elif corruption_type == "wrong_start":
            return (f"No. Error: wrong_start. "
                    f"Procedure should begin with '{corruption_detail['correct_first']}', "
                    f"not '{corruption_detail['wrong_first']}'.")

    #cascading errors after the first wrong step get a plain No
    return "No"


def build_prm_record(procedure: str, available_actions: list[str],
                     steps_so_far: list[str], candidate: str, label: int,
                     step_index: int = 0, corruption_type=None, corruption_detail=None) -> dict:
    #steps_so_far is empty at the beginning of a procedure — the agent starts fresh
    #as the procedure progresses, completed actions accumulate here
    if steps_so_far:
        steps_str = " → ".join(steps_so_far)
    else:
        steps_str = "(none)"

    actions_str = " | ".join(available_actions)

    user_content = (
        f"Procedure: {procedure}\n\n"
        f"Available actions: {actions_str}\n\n"
        f"Steps completed so far: {steps_str}\n\n"
        f"Proposed next action: {candidate}\n\n"
        f"Is this the correct next step?"
    )

    answer = generate_answer(label, step_index, corruption_type, corruption_detail)

    return {
        "messages": [
            {"role": "system",    "content": PRM_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


#build a lookup from file_index to the full flat list of action names
#we use the extracted workflow not ground truth on purpose — the end-to-end
#pipeline (including the extractor) is what gets evaluated
def build_action_lists(predictions: list[dict]) -> dict[int, list[str]]:
    action_lists = {}
    for pred in predictions:
        wf = pred.get("workflow") or {}
        action_lists[pred["file_index"]] = [a["name"] for a in wf.get("actions", [])]
    return action_lists


#the (file_index, steps_so_far, candidate) of every step a positive trace labels Yes.
#a negative whose first-wrong step has the same key is an unlearnable contradiction (the identical
#prompt is a valid Yes on another path), so we drop it — branches that share a prefix and a valid
#next action are indistinguishable to the PRM, which never sees the gateway condition that separates them
def _positive_keys(traces: list[dict], action_lists: dict[int, list[str]]) -> set:
    keys = set()
    for trace in traces:
        detail = trace.get("corruption_detail")
        if detail and "first_wrong_step" in detail:
            continue  # negative trace
        if not action_lists.get(trace["file_index"]):
            continue
        steps = trace["steps"]
        for i in range(len(steps)):
            prefix = tuple(s["action"] for s in steps[:i])
            keys.add((trace["file_index"], prefix, steps[i]["action"]))
    return keys


#expand trace-level records into step-level SFT examples, dropping any over MAX_TOKENS.
#returns (examples, kept, dropped, contradictions). shared by the dedup variant so the rules stay in one place.
def expand_to_sft(traces: list[dict], predictions: list[dict]) -> tuple[list[dict], int, int, int]:
    action_lists = build_action_lists(predictions)
    positive_keys = _positive_keys(traces, action_lists)

    kept = dropped = contradictions = 0
    examples = []

    for trace in traces:
        fidx      = trace["file_index"]
        procedure = trace["procedure"]
        steps     = trace["steps"]

        available = action_lists.get(fidx, [])
        if not available:
            #no workflow found for this file_index, skip
            dropped += len(steps)
            continue

        corruption_type   = trace.get("corruption_type")
        corruption_detail = trace.get("corruption_detail")

        #for corrupted traces, only emit the labeled-wrong step (one No example per trace)
        #the leading Yes steps in a corrupted trace are byte-identical to the positive path's Yes steps
        #emitting them duplicates positives and inflates dataset size for no signal
        #for positive traces, emit every step as Yes (this is where all positives come from)
        if corruption_detail and "first_wrong_step" in corruption_detail:
            indices_to_emit = [corruption_detail["first_wrong_step"]]
        else:
            indices_to_emit = list(range(len(steps)))

        for i in indices_to_emit:
            step         = steps[i]
            steps_so_far = [s["action"] for s in steps[:i]]

            #drop No examples that contradict a Yes elsewhere (same procedure, history, candidate)
            if step["label"] == 0 and (fidx, tuple(steps_so_far), step["action"]) in positive_keys:
                contradictions += 1
                continue

            record = build_prm_record(
                procedure, available, steps_so_far, step["action"], step["label"],
                step_index=i, corruption_type=corruption_type, corruption_detail=corruption_detail,
            )

            if estimate_tokens(record) <= MAX_TOKENS:
                examples.append(record)
                kept += 1
            else:
                dropped += 1

    return examples, kept, dropped, contradictions


def main():
    with open(PRM_DATA_PATH, encoding="utf-8") as f:
        traces = json.load(f)

    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        predictions = json.load(f)

    examples, kept, dropped, contradictions = expand_to_sft(traces, predictions)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    #count answers — every negative now starts with "No. Error: ..." after cascade cut,
    #so we match any answer beginning with "No" to capture both plain "No" and explained negatives
    n_yes = sum(1 for e in examples if e["messages"][-1]["content"] == "Yes")
    n_no  = sum(1 for e in examples if e["messages"][-1]["content"].startswith("No"))
    n_no_explained = sum(1 for e in examples if e["messages"][-1]["content"].startswith("No. Error: "))
    n_no_plain     = n_no - n_no_explained

    print(f"Traces processed : {len(traces)}")
    print(f"Step examples    : kept={kept}  dropped={dropped}  contradictions_filtered={contradictions}")
    print(f"  Yes (correct)        : {n_yes}")
    print(f"  No  (wrong)          : {n_no}")
    print(f"    with explanation   : {n_no_explained}")
    print(f"    plain (cascade)    : {n_no_plain}")
    print(f"  Ratio                : 1 : {n_no / max(n_yes, 1):.2f}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
