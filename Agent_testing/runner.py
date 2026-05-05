
#we run each agent on every predicted branch of every held-out procedure and save the picks
#paths come from the EXTRACTED graph so this
#evaluates the end-to-end pipeline 


import json
from dataclasses import dataclass
from pathlib import Path


class PlanningAgent:

    #predicted_states and id_to_name are optional and only used by agents that consult the
    #extracted graph as a tool 
    def pick(self, procedure_text: str, completed_names: list[str],
             candidate_names: list[str] | None,
             predicted_states: list[dict] | None = None,
             id_to_name: dict[str, str] | None = None) -> tuple[str, dict]:
        raise NotImplementedError


@dataclass
class ProcedureCase:
    #one held-out procedure plus the predicted graph the extractor produced
    file_index: int
    procedure_text: str
    pred_action_names: list[str]            #candidates for methods 2/3
    pred_id_to_name: dict[str, str]         #predicted action ID transformed to act name
    pred_execution_states: list[dict]       #predicted execution graph 

    @classmethod
    def build(cls, held_out_record: dict, prediction_record: dict) -> "ProcedureCase":
        wf = prediction_record.get("workflow") or {}
        pred_actions = wf.get("actions") or []
        return cls(
            file_index=held_out_record["file_index"],
            procedure_text=held_out_record["procedure_text"],
            pred_action_names=[a["name"] for a in pred_actions],
            pred_id_to_name={a["id"]: a["name"] for a in pred_actions},
            pred_execution_states=prediction_record.get("execution_states") or [],
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
            #a terminal state with no outgoing options is a finished path
            paths.append(list(prefix))
            if not nexts:
                return

        for nxt in sorted(nexts):
            walk(prefix + (nxt,), depth + 1)

    walk((), 0)
    return paths


def _resolve_picked_id(picked: str, name_to_id: dict[str, str],
                       pred_action_names: list[str]) -> str | None:
    #map the agent's pick (a name string) back to a predicted action ID.
    #exact match first; fall back to case-insensitive substring match for free-form output
    #from llama_bare. returns None if nothing matches — the agent invented something.
    if picked in name_to_id:
        return name_to_id[picked]
    pl = picked.lower().strip().rstrip(".")
    for name in pred_action_names:
        nl = name.lower()
        if nl == pl or nl in pl or pl.startswith(nl):
            return name_to_id[name]
    return None


#fields from the agent info dict that we actually surface in the saved JSON.
#tuples (not sets) so insertion order in the saved JSON is stable and readable.
#raw pre-softmax scores (prm_scores, llm_scores) and llm_temp are dropped — the
#normalised distributions and final blend are enough for manual review.
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
              max_steps: int = 20, give_candidates: bool = True) -> dict:
    #the agent picks freely at every step. we walk the predicted execution graph along
    #the agent's actual choices and record what was valid at each point.
    #at the end, match the resulting trajectory to the enumerated gold paths so we know
    #which branch (if any) the agent ended up committing to.
    candidates = case.pred_action_names if give_candidates else None
    if give_candidates and not candidates:
        return {"status": "no_candidates", "matched_branch": None,
                "match_type": "off_path", "completed_trajectory": [], "steps": []}

    states_by_prefix: dict[tuple, list[dict]] = {}
    for s in case.pred_execution_states:
        states_by_prefix.setdefault(tuple(s["completed_actions"]), []).append(s)

    name_to_id = {v: k for k, v in case.pred_id_to_name.items()}

    completed_ids: list[str] = []
    steps: list[dict] = []
    status = "max_steps"

    for step_idx in range(max_steps):
        states = states_by_prefix.get(tuple(completed_ids), [])
        if not states:
            #should only happen if the agent's previous pick was somehow accepted into a
            #prefix that doesn't exist in the graph — defensive
            status = "off_path"
            break

        valid_next: set[str] = set()
        terminal = False
        for s in states:
            valid_next.update(s["available_next"])
            if s.get("can_terminate"):
                terminal = True
        #all distinct condition combinations consistent with the agent's history so far
        #(read-only — for manual review of the trace, doesn't affect the rollout)
        conditions_active = sorted({tuple(s.get("conditions_met") or []) for s in states})

        if terminal and not valid_next:
            status = "completed"
            break

        completed_names = [case.pred_id_to_name.get(aid, aid) for aid in completed_ids]
        valid_next_names = [case.pred_id_to_name.get(aid, aid) for aid in valid_next]

        picked, info = agent.pick(
            case.procedure_text, completed_names, candidates,
            predicted_states=case.pred_execution_states,
            id_to_name=case.pred_id_to_name,
        )
        picked_id = _resolve_picked_id(picked, name_to_id, case.pred_action_names)
        is_valid = picked_id is not None and picked_id in valid_next

        #lay out the step record so the eval-relevant fields are at the top:
        #  step → context → what was valid → what the agent picked → was it correct → tool fired
        #scoring detail goes into a "scores" sub-dict, tool detail into "tool" — both at the bottom
        step_dict: dict = {
            "step": step_idx + 1,
            "completed_before": list(completed_names),
            "conditions_active": [list(c) for c in conditions_active],
            "valid_options": valid_next_names,
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

        completed_ids.append(picked_id)

    #figure out which gold path (if any) the agent's trajectory matches
    gold_paths_ids = enumerate_paths(case.pred_execution_states)
    matched_branch: int | None = None
    match_type = "off_path"
    for i, path in enumerate(gold_paths_ids):
        if completed_ids == path:
            matched_branch, match_type = i, "exact"
            break
    if matched_branch is None:
        for i, path in enumerate(gold_paths_ids):
            if 0 < len(completed_ids) < len(path) and path[: len(completed_ids)] == completed_ids:
                matched_branch, match_type = i, "prefix"
                break

    completed_trajectory = [case.pred_id_to_name.get(aid, aid) for aid in completed_ids]

    return {
        "status": status,                          # completed | off_path | max_steps
        "matched_branch": matched_branch,          # gold-path index or null
        "match_type": match_type,                  # exact | prefix | off_path
        "completed_trajectory": completed_trajectory,
        "steps": steps,
    }


def run_inference(cases: list[ProcedureCase], agent: PlanningAgent,
                  give_candidates: bool = True, max_steps: int = 20) -> list[dict]:
    out = []
    for i, case in enumerate(cases):
        gold_paths_ids = enumerate_paths(case.pred_execution_states)
        gold_branches = [
            {"branch": idx, "path": [case.pred_id_to_name.get(aid, aid) for aid in p]}
            for idx, p in enumerate(gold_paths_ids)
        ]
        print(f"[{i+1}/{len(cases)}] file_index={case.file_index} "
              f"({len(case.pred_action_names)} pred actions, {len(gold_branches)} gold paths)")
        rollout = walk_free(case, agent, max_steps=max_steps, give_candidates=give_candidates)
        out.append({
            "file_index": case.file_index,
            "procedure_text": case.procedure_text,
            "predicted_actions": case.pred_action_names if give_candidates else None,
            "gold_branches": gold_branches,
            "rollout": rollout,
        })
    return out


def load_cases(held_out_path: Path, predictions_path: Path) -> list[ProcedureCase]:
    #held_out gives us procedure_text and the file_index 
    #predictions gives us predicted actions AND the execution graph for path enumeration
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
