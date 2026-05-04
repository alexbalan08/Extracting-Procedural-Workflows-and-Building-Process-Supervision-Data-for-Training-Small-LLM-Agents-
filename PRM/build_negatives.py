
#Each records execution_states are split into individual execution paths
#and converted to flat sequences of human-readable action names.

#we will do skip_action which drops an action and from that point we have label 0
#swap_adjacent where we swap two consecutive actions
#wrong_branch at a split we continue with actions from the wrong branch (the wrong condition)
#premature_stop just fiish faster all steps are correct but can_terminate will be null
#repeat_completed replaces a step with a previously completed action — teaches the PRM not to score
#  already-done actions highly (failure mode seen in held-out: PRM picks the very first action again at step 2)
#wrong_start replaces the very first action with a wrong one — currently no corruption produces a step-0
#  negative so the PRM defaults to ~0.99 confidence on every candidate at trajectory start

#we need to show the corruption type and the step so the traces are easy to follow by non experts humans

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


#the extractor stores actions by ID internally but training traces need human readable names
#this just builds a lookup dict so we can go from ID to name anywhere in the pipeline
def build_action_map(workflow):
    return {a["id"]: a["name"] for a in workflow.get("actions", [])}


#execution_states from the graph traversal contains ALL states from ALL branches into one list
#this function separates them back into individual paths, one path per terminal state (can_terminate=True)
def split_into_paths(states):

    terminals = [s for s in states if s.get("can_terminate")]
    if not terminals:
        return []

    paths = []
    for term in terminals:
        term_actions = term["completed_actions"]
        term_conds = set(term.get("conditions_met", []))

        path = []
        for s in states:
            s_actions = s.get("completed_actions", [])
            s_conds = set(s.get("conditions_met", []))
            #if completed actions and conditions met are included in the terminal dtates for that branch 
            if s_actions == term_actions[: len(s_actions)] and s_conds <= term_conds:
                path.append(s)

        path.sort(key=lambda s: len(s.get("completed_actions", [])))
        paths.append(path)

    return paths




#state_0: completed=[],                        conditions={}
#state_1: completed=["action_1"],              conditions={}
#state_2: completed=["action_1","action_2"],   conditions={"approved"}
#produces: [{"action":"verify document","label":1}, {"action":"approve payment","label":1,"condition_reached":["approved"]}]
#we start at index 1 sicne 0 is always start node
def path_to_steps(path_states, action_map):

    steps = []
    prev_conds = set()

    for i in range(1, len(path_states)):
        completed = path_states[i].get("completed_actions", [])
        if not completed:
            continue
        action_id = completed[-1]
        action_name = action_map.get(action_id, action_id)

        curr_conds = set(path_states[i].get("conditions_met", []))
        new_conds = curr_conds - prev_conds
        prev_conds = curr_conds

        step = {"action": action_name, "label": 1}
        if new_conds:
            step["condition_reached"] = sorted(new_conds)
        steps.append(step)

    return steps




#removes one intermediate action from the trace but never the first or last
#everything from the removal point onward gets label=0 because the sequence is now wrong as we ve seenn in normal prm literature
#the PRM needs to learn that skipping a required step makes all subsequent steps wrong even if they look right
#we need at least 3 steps so there is at least one intermediate step to skip
def corrupt_skip_action(steps):
    if len(steps) < 3:
        return None

    pos = random.randint(1, len(steps) - 2)
    corrupted = copy.deepcopy(steps)
    skipped = corrupted.pop(pos)

    for s in corrupted[pos:]:
        s["label"] = 0

    return corrupted, {
        "skipped_action": skipped["action"],
        "at_position": pos,
        "first_wrong_step": pos,
    }


#swaps two consecutive intermediate actions, wrong order is a common agent mistake especially around gateways
#everything from the swap point onward gets label=0 because the ordering is now wrong
#we need at least 4 steps so there are at least two intermediate steps that can be swapped
# upper bound is len-3 we never want to swap last 3 without the fort 
def corrupt_swap_adjacent(steps):
    if len(steps) < 4:
        return None

    pos = random.randint(1, len(steps) - 3)
    corrupted = copy.deepcopy(steps)
    corrupted[pos], corrupted[pos + 1] = corrupted[pos + 1], corrupted[pos]

    for s in corrupted[pos:]:
        s["label"] = 0

    return corrupted, {
        "swapped": [steps[pos]["action"], steps[pos + 1]["action"]],
        "at_position": pos,
        "first_wrong_step": pos,
    }


#this one requires multiple paths to work and it simulates the agent taking the wrong branch at a gateway
#we find where the current path and another path first diverge then continue with the wrong path s actions from there
#this does not work for linear workflow with only one path
def corrupt_wrong_branch(steps, all_paths_steps, path_idx):
    """After a shared prefix, continue with actions from a different branch."""
    current = steps

    #shuffle so calling this multiple times per path picks different "other" branches
    other_indices = [i for i in range(len(all_paths_steps)) if i != path_idx]
    random.shuffle(other_indices)

    for other_idx in other_indices:
        other = all_paths_steps[other_idx]

        #to find where the two paths diverge
        diverge_at = None
        for j in range(min(len(current), len(other))):
            if current[j]["action"] != other[j]["action"]:
                diverge_at = j
                break

        if diverge_at is None or diverge_at == 0:
            continue

        #currennt correct path + continuation from wrong path
        corrupted = copy.deepcopy(current[:diverge_at])
        wrong_tail = copy.deepcopy(other[diverge_at:])
        for s in wrong_tail:
            s["label"] = 0
        corrupted.extend(wrong_tail)

        return corrupted, {
            "correct_next": current[diverge_at]["action"],
            "wrong_next": other[diverge_at]["action"],
            "diverge_at": diverge_at,
            "first_wrong_step": diverge_at,
        }

    return None


#replaces a step with an action that was already completed earlier in the same trace
#cascades label=0 from that point on — the action is wrong and any continuation off this branch is also wrong
#we need at least 3 steps so there is a valid pos with at least one different prior action to repeat
def corrupt_repeat_completed(steps):
    if len(steps) < 3:
        return None

    pos = random.randint(1, len(steps) - 1)
    prior_actions = [s["action"] for s in steps[:pos]]
    #must pick a prior action that isn't already what this step is — otherwise it's not a corruption
    candidates = [a for a in prior_actions if a != steps[pos]["action"]]
    if not candidates:
        return None

    repeated = random.choice(candidates)
    corrupted = copy.deepcopy(steps)
    corrupted[pos]["action"] = repeated

    for s in corrupted[pos:]:
        s["label"] = 0

    return corrupted, {
        "repeated_action": repeated,
        "at_position": pos,
        "first_wrong_step": pos,
    }


#replaces step 0 with a wrong opening action — teaches the PRM to reject incorrect first steps
#without this, history=empty has only positive examples in training data and the PRM scores every
#candidate at ~0.99 at trajectory start (seen in held-out: all four candidates >0.99 at step 1)
#we need the workflow's full action set so we can pick a non-first action to substitute in
def corrupt_wrong_start(steps, all_action_names):
    if len(steps) < 2:
        return None

    correct_first = steps[0]["action"]
    wrong_options = [a for a in all_action_names if a != correct_first]
    if not wrong_options:
        return None

    wrong_first = random.choice(wrong_options)
    corrupted = copy.deepcopy(steps)
    corrupted[0]["action"] = wrong_first

    for s in corrupted:
        s["label"] = 0

    return corrupted, {
        "wrong_first":   wrong_first,
        "correct_first": correct_first,
        "first_wrong_step": 0,
    }


#truncates the trace at a random point before the end
#this teaches the PRM that a trace can have all green steps and still be wrong if it stopped too early
#lower bound is 2 so we always keep at least two steps and bcs of this we need at least 3 steps so stop_at can be strictly less than the full length
def corrupt_premature_stop(steps):
    """Truncate the trace early.  All present steps are individually correct."""
    if len(steps) < 3:
        return None

    stop_at = random.randint(2, len(steps) - 1)
    corrupted = copy.deepcopy(steps[:stop_at])
    #last step gets label=0 as a "wrong termination" signal for step-level PRM training
    corrupted[-1]["label"] = 0

    return corrupted, {
        "stopped_at": stop_at,
        "full_length": len(steps),
        "first_wrong_step": stop_at - 1,
    }



#i fixed a seed for reproductibility
#we want to have for each positive flow 4 negatives
def main():
    _here = Path(__file__).parent

    ap = argparse.ArgumentParser(description="Build step-labeled traces for PRM training")
    ap.add_argument("--input",  type=Path,
                    default=_here.parent / "Extraction_results" / "extraction_predictions.json")
    ap.add_argument("--output", type=Path,
                    default=_here / "prm_training_data.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_corruption", type=int, default=1,
                    help="negatives per corruption type per path")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    all_examples = []

    for record in records:
        states   = record.get("execution_states", [])
        workflow  = record.get("workflow", {})
        action_map = build_action_map(workflow)
        all_action_names = list(action_map.values())

        paths = split_into_paths(states)
        if not paths:
            continue

        #convert all paths to step lists upfront so wrong_branch can compare across paths
        all_paths_steps = [path_to_steps(p, action_map) for p in paths]

        for path_idx, steps in enumerate(all_paths_steps):
            if not steps:
                continue

            # ── positive ──
            #deepcopy so any later corruption on the same steps list doesnt mutate the positive
            all_examples.append({
                "file_index":        record["file_index"],
                "procedure":         record["procedure_text"],
                "steps":             copy.deepcopy(steps),
                "complete":          True,
                "label":             1,
                "corruption_type":   None,
                "corruption_detail": None,
            })

            # ── negatives ──
            #we attempt all four corruption types and skip any that return None (path too short)
            #complete is True for all corruptions except premature_stop since those traces do reach an end
            #just the wrong one, whereas premature_stop never reaches any terminal state at all
            for _ in range(args.n_per_corruption):
                #premature_stop dropped — it labeled the same (history, candidate) that exists
                #as a positive elsewhere as No, creating contradictory labels the PRM cannot learn from.
                #termination decisions belong in a trajectory-level outcome verifier, not the step PRM.
                attempts = [
                    ("skip_action",      corrupt_skip_action(steps)),
                    ("swap_adjacent",    corrupt_swap_adjacent(steps)),
                    ("wrong_branch",     corrupt_wrong_branch(steps, all_paths_steps, path_idx)),
                    ("repeat_completed", corrupt_repeat_completed(steps)),
                    ("wrong_start",      corrupt_wrong_start(steps, all_action_names)),
                ]
                for ctype, result in attempts:
                    if result is None:
                        continue
                    corrupted_steps, detail = result
                    all_examples.append({
                        "file_index":        record["file_index"],
                        "procedure":         record["procedure_text"],
                        "steps":             corrupted_steps,
                        "complete":          ctype != "premature_stop",
                        "label":             0,
                        "corruption_type":   ctype,
                        "corruption_detail": detail,
                    })

    n_pos = sum(1 for e in all_examples if e["label"] == 1)
    n_neg = sum(1 for e in all_examples if e["label"] == 0)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"Records processed : {len(records)}")
    print(f"Paths (positives) : {n_pos}")
    print(f"Negatives         : {n_neg}")
    #ideally around 4:1 but shorter traces produce fewer negatives since corruption functions return None
    print(f"  ratio           : 1 : {n_neg / max(n_pos, 1):.1f}")

    print()
    types = Counter(e["corruption_type"] for e in all_examples if e["label"] == 0)
    for t, c in sorted(types.items()):
        print(f"  {t:<20} {c}")
    print()
    print(f"Saved {len(all_examples)} total examples to {args.output}")


if __name__ == "__main__":
    main()