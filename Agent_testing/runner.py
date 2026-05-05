
#we run each agent on every held-out procedure and save the picks.
#paths can come from either the EXTRACTED graph (mode="predicted") or the GOLD
#workflow in held_out.json (mode="gold"). running both lets us separate
#"agent quality" (gold mode) from "extraction quality" (predicted mode).


import json
from dataclasses import dataclass
from pathlib import Path


class PlanningAgent:

    #predicted_states and id_to_name are optional and only used by agents that consult
    #the graph as a tool. they receive whichever graph is active for the eval (predicted or gold).
    def pick(self, procedure_text: str, completed_names: list[str],
             candidate_names: list[str] | None,
             predicted_states: list[dict] | None = None,
             id_to_name: dict[str, str] | None = None) -> tuple[str, dict]:
        raise NotImplementedError


@dataclass
class ProcedureCase:
    #one held-out procedure with both the predicted graph (from the extractor) and
    #the gold graph (from held_out.json). walk_free picks one of the two at eval time.
    file_index: int
    procedure_text: str
    pred_action_names: list[str]
    pred_id_to_name: dict[str, str]
    pred_execution_states: list[dict]
    gold_action_names: list[str]
    gold_id_to_name: dict[str, str]
    gold_execution_states: list[dict]

    @classmethod
    def build(cls, held_out_record: dict, prediction_record: dict) -> "ProcedureCase":
        pred_wf = prediction_record.get("workflow") or {}
        pred_actions = pred_wf.get("actions") or []
        gold_wf = held_out_record.get("workflow") or {}
        gold_actions = gold_wf.get("actions") or []
        return cls(
            file_index=held_out_record["file_index"],
            procedure_text=held_out_record["procedure_text"],
            pred_action_names=[a["name"] for a in pred_actions],
            pred_id_to_name={a["id"]: a["name"] for a in pred_actions},
            pred_execution_states=prediction_record.get("execution_states") or [],
            gold_action_names=[a["name"] for a in gold_actions],
            gold_id_to_name={a["id"]: a["name"] for a in gold_actions},
            gold_execution_states=held_out_record.get("execution_states") or [],
        )

    def graph(self, mode: str) -> tuple[list[str], dict[str, str], list[dict]]:
        #returns (action_names, id_to_name, execution_states) for the requested mode
        if mode == "gold":
            return self.gold_action_names, self.gold_id_to_name, self.gold_execution_states
        return self.pred_action_names, self.pred_id_to_name, self.pred_execution_states


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
                       action_names: list[str]) -> str | None:
    #map the agent's pick (a name string) back to an action ID in the active graph.
    #exact match first; fall back to case-insensitive substring match for free-form output
    #from llama_bare. returns None if nothing matches — the agent invented something.
    if picked in name_to_id:
        return name_to_id[picked]
    pl = picked.lower().strip().rstrip(".")
    for name in action_names:
        nl = name.lower()
        if nl == pl or nl in pl or pl.startswith(nl):
            return name_to_id[name]
    return None


#fields from the agent info dict that we surface in the saved JSON.
#tuples (not sets) so insertion order is stable. raw pre-softmax scores and llm_temp
#are dropped — the normalised distributions and final blend are enough for review.
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
    #the agent picks freely at every step. we walk the chosen execution graph along
    #the agent's actual choices and record what was valid at each point.
    #at the end, match the resulting trajectory to the enumerated paths in that graph
    #to know which branch (if any) the agent ended up committing to.
    #
    #mode = "predicted" → uses extractor output. tests end-to-end pipeline.
    #mode = "gold"      → uses ground truth. tests the agent in isolation.
    action_names, id_to_name, execution_states = case.graph(mode)

    candidates = action_names if give_candidates else None
    if give_candidates and not candidates:
        return {"status": "no_candidates", "matched_branch": None,
                "match_type": "off_path", "completed_trajectory": [], "steps": []}

    states_by_prefix: dict[tuple, list[dict]] = {}
    for s in execution_states:
        states_by_prefix.setdefault(tuple(s["completed_actions"]), []).append(s)

    name_to_id = {v: k for k, v in id_to_name.items()}

    completed_ids: list[str] = []
    steps: list[dict] = []
    status = "max_steps"

    for step_idx in range(max_steps):
        states = states_by_prefix.get(tuple(completed_ids), [])
        if not states:
            status = "off_path"
            break

        valid_next: set[str] = set()
        terminal = False
        for s in states:
            valid_next.update(s["available_next"])
            if s.get("can_terminate"):
                terminal = True
        conditions_active = sorted({tuple(s.get("conditions_met") or []) for s in states})

        if terminal and not valid_next:
            status = "completed"
            break

        completed_names = [id_to_name.get(aid, aid) for aid in completed_ids]
        valid_next_names = [id_to_name.get(aid, aid) for aid in valid_next]

        picked, info = agent.pick(
            case.procedure_text, completed_names, candidates,
            predicted_states=execution_states,
            id_to_name=id_to_name,
        )
        picked_id = _resolve_picked_id(picked, name_to_id, action_names)
        is_valid = picked_id is not None and picked_id in valid_next

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

    #figure out which path (if any) the agent's trajectory matches in the active graph
    paths_ids = enumerate_paths(execution_states)
    matched_branch: int | None = None
    match_type = "off_path"
    for i, path in enumerate(paths_ids):
        if completed_ids == path:
            matched_branch, match_type = i, "exact"
            break
    if matched_branch is None:
        for i, path in enumerate(paths_ids):
            if 0 < len(completed_ids) < len(path) and path[: len(completed_ids)] == completed_ids:
                matched_branch, match_type = i, "prefix"
                break

    completed_trajectory = [id_to_name.get(aid, aid) for aid in completed_ids]

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
        action_names, id_to_name, execution_states = case.graph(mode)
        paths_ids = enumerate_paths(execution_states)
        branches = [
            {"branch": idx, "path": [id_to_name.get(aid, aid) for aid in p]}
            for idx, p in enumerate(paths_ids)
        ]
        print(f"[{i+1}/{len(cases)}] file_index={case.file_index} "
              f"({len(action_names)} actions, {len(branches)} branches, mode={mode})")
        rollout = walk_free(case, agent, max_steps=max_steps,
                            give_candidates=give_candidates, mode=mode)
        out.append({
            "file_index": case.file_index,
            "eval_mode": mode,
            "procedure_text": case.procedure_text,
            "candidate_actions": action_names if give_candidates else None,
            "branches": branches,
            "rollout": rollout,
        })
    return out


def load_cases(held_out_path: Path, predictions_path: Path) -> list[ProcedureCase]:
    #held_out gives us procedure_text, the file_index keyspace, AND the gold graph.
    #predictions gives us the predicted actions and predicted execution graph.
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
