
#we run each agent on every predicted branch of every held-out procedure and save the picks
#paths come from the EXTRACTED graph (extraction_predictions.json execution_states) so this
#evaluates the end-to-end pipeline — exactly what prepare_prm_data.py does at training time


import json
from dataclasses import dataclass
from pathlib import Path


class PlanningAgent:

    def pick(self, procedure_text: str, completed_names: list[str],
             candidate_names: list[str] | None) -> tuple[str, dict]:
        raise NotImplementedError


@dataclass
class ProcedureCase:
    #one held-out procedure plus the predicted graph the extractor produced
    file_index: int
    procedure_text: str
    pred_action_names: list[str]            #candidates for methods 2/3
    pred_id_to_name: dict[str, str]         #predicted action ID → predicted action name
    pred_execution_states: list[dict]       #predicted execution graph (action IDs)

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
    #walk the execution-state graph and return one action-id sequence per distinct path
    #from start to a can_terminate=True state. Loops are bounded because trace_builder
    #pre-enumerates a finite list of states upstream.
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
            #(if it also has options, callers can both stop here and continue — keep both)
            paths.append(list(prefix))
            if not nexts:
                return

        for nxt in sorted(nexts):
            walk(prefix + (nxt,), depth + 1)

    walk((), 0)
    return paths


def walk_trajectories(case: ProcedureCase, agent: PlanningAgent,
                      give_candidates: bool = True) -> list[dict]:
    #for each predicted path, ask the agent at every step what it would pick given the
    #path's history. We advance along the predicted path (teacher-forced) so each branch
    #produces its own clean trace — the agent's pick is recorded next to what the path expects.
    candidates = case.pred_action_names if give_candidates else None
    if give_candidates and not candidates:
        return []

    paths = enumerate_paths(case.pred_execution_states)
    if not paths:
        return []

    trajectories: list[dict] = []
    for branch_idx, path_ids in enumerate(paths):
        path_names = [case.pred_id_to_name.get(aid, aid) for aid in path_ids]
        completed_names: list[str] = []
        steps: list[dict] = []
        for step_idx, expected_name in enumerate(path_names):
            picked, info = agent.pick(case.procedure_text, completed_names, candidates)
            steps.append({
                "step": step_idx + 1,
                "completed_before": list(completed_names),
                "expected": expected_name,
                "picked": picked,
                **info,
            })
            #advance along the predicted path, NOT the agent's pick — keeps the branch clean
            completed_names.append(expected_name)
        trajectories.append({
            "branch": branch_idx,
            "path": path_names,
            "steps": steps,
        })
    return trajectories


def run_inference(cases: list[ProcedureCase], agent: PlanningAgent,
                  give_candidates: bool = True) -> list[dict]:
    out = []
    for i, case in enumerate(cases):
        n_paths = len(enumerate_paths(case.pred_execution_states))
        print(f"[{i+1}/{len(cases)}] file_index={case.file_index} "
              f"({len(case.pred_action_names)} pred actions, {n_paths} predicted paths)")
        trajectories = walk_trajectories(case, agent, give_candidates)
        out.append({
            "file_index": case.file_index,
            "procedure_text": case.procedure_text,
            "predicted_actions": case.pred_action_names if give_candidates else None,
            "trajectories": trajectories,
        })
    return out


def load_cases(held_out_path: Path, predictions_path: Path) -> list[ProcedureCase]:
    #held_out: gives us procedure_text and the file_index keyspace
    #predictions: gives us predicted actions AND the execution-state graph for path enumeration
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
