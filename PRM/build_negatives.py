"""
build_negatives.py  –  Generate step-level labeled traces for PRM training.

Each record's execution_states are split into individual execution paths
and converted to flat sequences of human-readable action names.

Corruption strategies (each produces step-level labels):
  skip_action      – drop an intermediate action; steps from that point on are label=0
  swap_adjacent    – swap two consecutive actions; steps from that point on are label=0
  wrong_branch     – at a fork, continue with actions from the wrong path; label=0 from divergence
  premature_stop   – truncate the trace early; all present steps are correct but complete=false

Output schema per example:
  {
    "file_index":        int,
    "procedure":         str,
    "steps":             [{"action": str, "label": 1|0, "condition_reached": [str]?}],
    "complete":          bool,
    "label":             1 | 0,
    "corruption_type":   str | null,
    "corruption_detail": dict | null
  }
"""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_action_map(workflow):
    """action_id -> human-readable name."""
    return {a["id"]: a["name"] for a in workflow.get("actions", [])}


def split_into_paths(states):
    """Split execution_states into individual paths (one per terminal state)."""
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
            # s belongs to this path if its completed_actions is a prefix and
            # its conditions are a subset (empty conditions match any branch)
            if s_actions == term_actions[: len(s_actions)] and s_conds <= term_conds:
                path.append(s)

        path.sort(key=lambda s: len(s.get("completed_actions", [])))
        paths.append(path)

    return paths


def path_to_steps(path_states, action_map):
    """Convert a path (ordered states) into a list of step dicts."""
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


# ── Corruption functions ─────────────────────────────────────────────────────
# Each returns (corrupted_steps, detail_dict) or None.

def corrupt_skip_action(steps):
    """Remove one intermediate action; everything from that point is wrong."""
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


def corrupt_swap_adjacent(steps):
    """Swap two consecutive intermediate actions; wrong from swap point on."""
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


def corrupt_wrong_branch(steps, all_paths_steps, path_idx):
    """After a shared prefix, continue with actions from a different branch."""
    current = steps

    for other_idx, other in enumerate(all_paths_steps):
        if other_idx == path_idx:
            continue

        # Find where the two paths diverge
        diverge_at = None
        for j in range(min(len(current), len(other))):
            if current[j]["action"] != other[j]["action"]:
                diverge_at = j
                break

        if diverge_at is None or diverge_at == 0:
            continue

        # Prefix from current path (correct) + continuation from wrong path
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


def corrupt_premature_stop(steps):
    """Truncate the trace early.  All present steps are individually correct."""
    if len(steps) < 3:
        return None

    stop_at = random.randint(2, len(steps) - 1)
    corrupted = copy.deepcopy(steps[:stop_at])
    # steps themselves remain label=1 (each was correct in isolation)

    return corrupted, {
        "stopped_at": stop_at,
        "full_length": len(steps),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

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

        paths = split_into_paths(states)
        if not paths:
            continue

        all_paths_steps = [path_to_steps(p, action_map) for p in paths]

        for path_idx, steps in enumerate(all_paths_steps):
            if not steps:
                continue

            # ── positive ──
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
            for _ in range(args.n_per_corruption):
                attempts = [
                    ("skip_action",    corrupt_skip_action(steps)),
                    ("swap_adjacent",  corrupt_swap_adjacent(steps)),
                    ("wrong_branch",   corrupt_wrong_branch(steps, all_paths_steps, path_idx)),
                    ("premature_stop", corrupt_premature_stop(steps)),
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
    print(f"  ratio           : 1 : {n_neg / max(n_pos, 1):.1f}")

    print()
    types = Counter(e["corruption_type"] for e in all_examples if e["label"] == 0)
    for t, c in sorted(types.items()):
        print(f"  {t:<20} {c}")
    print()
    print(f"Saved {len(all_examples)} total examples to {args.output}")


if __name__ == "__main__":
    main()
