
#we run each agent on every held-out procedure and save the picks.
#two evaluation modes — both validate against gold execution_states (ground truth):
#  mode = "gold"  → agent also sees the gold graph. measures the agent's ceiling
#                   (what's possible if extraction were perfect).
#  mode = "predicted" → agent sees the EXTRACTED graph (with its noise/hallucinations)
#                   but we still grade against gold. measures real deployment quality.
#the gap between gold and predicted is the cost of extraction errors.


import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "Extraction_results"))
from validate_results import match_actions  # noqa: E402


class PlanningAgent:

    #predicted_states and id_to_name are optional and only used by agents that consult
    #the graph as a tool. they receive whichever graph the agent sees this run (predicted
    #for predicted mode, gold for gold mode) — never the validation graph in predicted mode.
    def pick(self, procedure_text: str, completed_names: list[str],
             candidate_names: list[str] | None,
             predicted_states: list[dict] | None = None,
             id_to_name: dict[str, str] | None = None) -> tuple[str, dict]:
        raise NotImplementedError


@dataclass
class ProcedureCase:
    file_index: int
    procedure_text: str
    #predicted graph from the extractor — what the agent sees in predicted mode
    pred_action_names: list[str]
    pred_id_to_name: dict[str, str]
    pred_execution_states: list[dict]
    #gold graph from held_out.json — always used as the validation reference
    gold_action_names: list[str]
    gold_id_to_name: dict[str, str]
    gold_execution_states: list[dict]
    #pred-action-name → gold-action-id mapping (built with match_actions fuzzy matching).
    #lets predicted mode validate the agent's pick against the gold graph. missing keys =
    #the extractor invented an action that has no gold counterpart (a hallucination).
    pred_name_to_gold_id: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, held_out_record: dict, prediction_record: dict) -> "ProcedureCase":
        pred_wf = prediction_record.get("workflow") or {}
        pred_actions = pred_wf.get("actions") or []
        gold_wf = held_out_record.get("workflow") or {}
        gold_actions = gold_wf.get("actions") or []

        #match_actions returns (tp, fn, fp, {gt_idx: pred_idx}) — fuzzy alignment by name
        _, _, _, gt_to_pred = match_actions(gold_actions, pred_actions)
        pred_name_to_gold_id = {
            pred_actions[pred_i]["name"]: gold_actions[gt_i]["id"]
            for gt_i, pred_i in gt_to_pred.items()
        }

        return cls(
            file_index=held_out_record["file_index"],
            procedure_text=held_out_record["procedure_text"],
            pred_action_names=[a["name"] for a in pred_actions],
            pred_id_to_name={a["id"]: a["name"] for a in pred_actions},
            pred_execution_states=prediction_record.get("execution_states") or [],
            gold_action_names=[a["name"] for a in gold_actions],
            gold_id_to_name={a["id"]: a["name"] for a in gold_actions},
            gold_execution_states=held_out_record.get("execution_states") or [],
            pred_name_to_gold_id=pred_name_to_gold_id,
        )


def enumerate_paths(execution_states: list[dict], max_depth: int = 30) -> list[list[str]]:
    by_prefix: dict[tuple, list[dict]] = {}
    for s in execution_states:
        by_prefix.setdefault(tuple(s["completed_actions"]), []).append(s)

    paths: list[list[str]] = []

    def walk(prefix: tuple, depth: int):
        if depth > max_depth:
            return
        states = by_prefix.get(prefix)
        if not states:
            return

        nexts: set[str] = set()
        terminal = False
        for s in states:
            if s.get("can_terminate"):
                terminal = True
            nexts.update(s["available_next"])

        if terminal:
            paths.append(list(prefix))
            if not nexts:
                return

        for nxt in sorted(nexts):
            walk(prefix + (nxt,), depth + 1)

    walk((), 0)
    return paths


def _resolve_picked_id(picked: str, name_to_id: dict[str, str],
                       action_names: list[str]) -> str | None:
    #map the agent's pick (a name string) back to an action ID in the INPUT graph.
    #exact match first; fall back to case-insensitive substring match for free-form output
    #from llama_bare. returns None if nothing matches.
    if picked in name_to_id:
        return name_to_id[picked]
    pl = picked.lower().strip().rstrip(".")
    for name in action_names:
        nl = name.lower()
        if nl == pl or nl in pl or pl.startswith(nl):
            return name_to_id[name]
    return None


#fields from the agent info dict that we surface in the saved JSON.
_SCORE_FIELDS = (
    "alpha",
    "top_score", "margin",
    "final", "prm_dist", "llm_dist",
)
_TOOL_FIELDS = (
    "tool_threshold", "tool_margin",
    "valid_next", "narrowed",
    "tool_useful", "narrowed_final",
)


def walk_free(case: ProcedureCase, agent: PlanningAgent,
              max_steps: int = 20, give_candidates: bool = True,
              mode: str = "predicted") -> dict:
    #the agent picks freely. validation ALWAYS happens against the gold graph.
    #in gold mode, the agent also sees the gold graph as input.
    #in predicted mode, the agent sees the predicted graph (real deployment) but is graded
    #against gold — picks that don't correspond to any gold action count as off-path.
    if mode == "gold":
        input_actions = case.gold_action_names
        input_id_to_name = case.gold_id_to_name
        input_exec_states = case.gold_execution_states
    elif mode == "predicted":
        input_actions = case.pred_action_names
        input_id_to_name = case.pred_id_to_name
        input_exec_states = case.pred_execution_states
    else:
        raise ValueError(f"Unknown mode: {mode}")

    val_exec_states = case.gold_execution_states
    val_id_to_name = case.gold_id_to_name

    candidates = input_actions if give_candidates else None
    if give_candidates and not candidates:
        return {"status": "no_candidates", "matched_branch": None,
                "match_type": "off_path", "completed_trajectory": [], "steps": []}

    val_states_by_prefix: dict[tuple, list[dict]] = {}
    for s in val_exec_states:
        val_states_by_prefix.setdefault(tuple(s["completed_actions"]), []).append(s)

    input_name_to_id = {v: k for k, v in input_id_to_name.items()}

    #two parallel histories: what the agent sees (input names) and what we validate (gold ids)
    completed_input_names: list[str] = []
    completed_val_ids: list[str] = []
    steps: list[dict] = []
    status = "max_steps"

    for step_idx in range(max_steps):
        val_states = val_states_by_prefix.get(tuple(completed_val_ids), [])
        if not val_states:
            status = "off_path"
            break

        valid_next_val_ids: set[str] = set()
        terminal = False
        for s in val_states:
            valid_next_val_ids.update(s["available_next"])
            if s.get("can_terminate"):
                terminal = True
        conditions_active = sorted({tuple(s.get("conditions_met") or []) for s in val_states})

        if terminal and not valid_next_val_ids:
            status = "completed"
            break

        valid_next_val_names = [val_id_to_name.get(aid, aid) for aid in valid_next_val_ids]

        picked, info = agent.pick(
            case.procedure_text, completed_input_names, candidates,
            predicted_states=input_exec_states,
            id_to_name=input_id_to_name,
        )

        #resolve agent's pick to an INPUT id, then translate to a VAL (gold) id for the check
        picked_input_id = _resolve_picked_id(picked, input_name_to_id, input_actions)
        if mode == "gold":
            picked_val_id = picked_input_id  #same namespace
        else:  #predicted
            picked_input_name = input_id_to_name.get(picked_input_id) if picked_input_id else picked
            picked_val_id = case.pred_name_to_gold_id.get(picked_input_name)

        is_valid = picked_val_id is not None and picked_val_id in valid_next_val_ids

        step_dict: dict = {
            "step": step_idx + 1,
            "completed_before": list(completed_input_names),
            "conditions_active": [list(c) for c in conditions_active],
            "valid_options": valid_next_val_names,
            "picked": picked,
            "is_valid": is_valid,
        }
        if "tool_called" in info:
            step_dict["tool_called"] = info["tool_called"]
        step_dict["scores"] = {k: info[k] for k in _SCORE_FIELDS if k in info}
        if step_dict.get("tool_called"):
            step_dict["tool"] = {k: info[k] for k in _TOOL_FIELDS if k in info}
        steps.append(step_dict)

        if not is_valid:
            status = "off_path"
            break

        completed_input_names.append(picked)
        completed_val_ids.append(picked_val_id)

    #match the agent's trajectory (in gold ids) against enumerated gold paths
    gold_paths_ids = enumerate_paths(val_exec_states)
    matched_branch: int | None = None
    match_type = "off_path"
    for i, path in enumerate(gold_paths_ids):
        if completed_val_ids == path:
            matched_branch, match_type = i, "exact"
            break
    if matched_branch is None:
        for i, path in enumerate(gold_paths_ids):
            if 0 < len(completed_val_ids) < len(path) and path[: len(completed_val_ids)] == completed_val_ids:
                matched_branch, match_type = i, "prefix"
                break

    completed_trajectory = [val_id_to_name.get(aid, aid) for aid in completed_val_ids]

    return {
        "status": status,
        "matched_branch": matched_branch,
        "match_type": match_type,
        "completed_trajectory": completed_trajectory,
        "steps": steps,
    }


def run_inference(cases: list[ProcedureCase], agent: PlanningAgent,
                  give_candidates: bool = True, max_steps: int = 20,
                  mode: str = "predicted") -> list[dict]:
    out = []
    for i, case in enumerate(cases):
        input_actions = case.gold_action_names if mode == "gold" else case.pred_action_names
        gold_paths_ids = enumerate_paths(case.gold_execution_states)
        gold_branches = [
            {"branch": idx, "path": [case.gold_id_to_name.get(aid, aid) for aid in p]}
            for idx, p in enumerate(gold_paths_ids)
        ]
        print(f"[{i+1}/{len(cases)}] file_index={case.file_index} "
              f"({len(input_actions)} actions seen, {len(gold_branches)} gold branches, mode={mode})")
        rollout = walk_free(case, agent, max_steps=max_steps,
                            give_candidates=give_candidates, mode=mode)
        out.append({
            "file_index": case.file_index,
            "eval_mode": mode,
            "procedure_text": case.procedure_text,
            "candidate_actions": input_actions if give_candidates else None,
            "branches": gold_branches,
            "rollout": rollout,
        })
    return out


def load_cases(held_out_path: Path, predictions_path: Path) -> list[ProcedureCase]:
    #held_out gives us procedure_text, file_index, AND the gold graph (validation source).
    #predictions gives us the predicted graph the agent sees in predicted mode.
    with open(held_out_path, encoding="utf-8") as f:
        held_out = json.load(f)
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    pred_by_idx = {p["file_index"]: p for p in predictions}

    cases = []
    skipped = 0
    for record in held_out:
        pred = pred_by_idx.get(record["file_index"])
        if pred is None or pred.get("workflow") is None or not pred.get("execution_states"):
            skipped += 1
            continue
        cases.append(ProcedureCase.build(record, pred))

    print(f"Loaded {len(cases)} cases (skipped {skipped} with no prediction / null workflow / no execution states)")
    return cases
