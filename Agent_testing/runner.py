
#we run each plaer on each held-out procedure and save the picked action
#trajectory for manual val

#one held-out procedure plus the action names the
#extractor predicted for it (the candidates the agent chooses from).


import json
from dataclasses import dataclass
from pathlib import Path


class PlanningAgent:

    def pick(self, procedure_text: str, completed_names: list[str],
             candidate_names: list[str] | None) -> tuple[str, dict]:
        raise NotImplementedError


@dataclass
class ProcedureCase:
    #One held-out procedure paired with the action list the extractor pipelie geerated
    #we will use the list of actios extracted to see if helps model to hallucinate less
    file_index: int
    procedure_text: str
    pred_action_names: list[str]   #extracted action

    @classmethod
    def build(cls, held_out_record: dict, prediction_record: dict) -> "ProcedureCase":
        #We pull the action names (not ids) from the predicted workflow because the PRM was trained on names 
        pred_actions = (prediction_record.get("workflow") or {}).get("actions") or []
        return cls(
            file_index=held_out_record["file_index"],
            procedure_text=held_out_record["procedure_text"],
            pred_action_names=[a["name"] for a in pred_actions],
        )


def walk_trajectory(case: ProcedureCase, agent: PlanningAgent, max_steps: int = 20,
                    give_candidates: bool = True) -> list[dict]:
    #Roll the agent forward for up to max_steps. The chosen action is appended to
    #`completed` and fed back on the next call. There is NO termination check —
    #the trace runs the full max_steps so the user sees what the model would do
    #(loops, repeats, dead-ends are all visible in the saved sequence).
    #
    #Worked example (max_steps=3, give_candidates=True):
    #   case.pred_action_names = ["Submit Form", "Review Form", "Approve Form"]
    #
    #   step 1: completed=[]                      → agent picks "Submit Form"
    #           trace[0] = {step:1, completed_before:[],            picked:"Submit Form"}
    #   step 2: completed=["Submit Form"]         → agent picks "Review Form"
    #           trace[1] = {step:2, completed_before:["Submit Form"], picked:"Review Form"}
    #   step 3: completed=["Submit Form","Review Form"] → agent picks "Approve Form"
    #           trace[2] = {step:3, completed_before:[...],          picked:"Approve Form"}
    #
    #   returns [trace[0], trace[1], trace[2]]

    #Method 1 (llama_bare) sets give_candidates=False so the agent generates free-form
    candidates = case.pred_action_names if give_candidates else None
    if give_candidates and not candidates:
        return []  #no extracted actions for this procedure, nothing to give the agent

    completed: list[str] = []
    trace: list[dict] = []
    for step_num in range(1, max_steps + 1):
        picked, info = agent.pick(case.procedure_text, completed, candidates)
        #Spread `info` into the step dict so PRM scores / raw responses appear inline
        trace.append({
            "step": step_num,
            "completed_before": list(completed),
            "picked": picked,
            **info,
        })
        completed.append(picked)
    return trace


def run_inference(cases: list[ProcedureCase], agent: PlanningAgent,
                  max_steps: int = 20, give_candidates: bool = True) -> list[dict]:
    #Run the agent on every case and return a list of saved-trace records.
    #
    #Output record shape (one per procedure):
    #   {
    #     "file_index": 1623832427,
    #     "procedure_text": "To start, reach out to the 1st Level Support...",
    #     "predicted_actions": ["Reach out to ...", "Provide feedback ...", ...],
    #     "picked_sequence":   ["Reach out to ...", "Provide feedback ...", ...],
    #     "steps": [
    #         { "step": 1, "completed_before": [], "picked": "...",
    #           "raw_response": "..."          # Llama agents
    #           "scores": {cand: P(yes), ...} # PRM agent
    #         },
    #         ...
    #     ]
    #   }
    out = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] file_index={case.file_index} "
              f"({len(case.pred_action_names)} pred actions)")
        trajectory = walk_trajectory(case, agent, max_steps, give_candidates)
        out.append({
            "file_index": case.file_index,
            "procedure_text": case.procedure_text,
            "predicted_actions": case.pred_action_names if give_candidates else None,
            #compact view first — just the picked sequence, ready for eyeballing
            "picked_sequence": [t["picked"] for t in trajectory],
            #full per-step record with diagnostics (raw model output / PRM scores)
            "steps": trajectory,
        })
    return out


def load_cases(held_out_path: Path, predictions_path: Path) -> list[ProcedureCase]:
    #Join held-out gold with extraction predictions on file_index. We need both:
    #  - held_out: gives us procedure_text and the file_index keyspace
    #  - predictions: gives us the predicted action names (the candidate list)
    #
    #Procedures with no matching prediction (or null workflow) are skipped — the
    #agent would have nothing to choose from in methods 2/3.
    #
    #Example join:
    #   held_out has: [{file_index: 100, procedure_text: "...", ...}, ...]
    #   predictions has: [{file_index: 100, workflow: {actions: [...]}}, ...]
    #   → ProcedureCase(file_index=100, procedure_text="...", pred_action_names=[...])
    with open(held_out_path, encoding="utf-8") as f:
        held_out = json.load(f)
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    pred_by_idx = {p["file_index"]: p for p in predictions}

    cases = []
    skipped = 0
    for record in held_out:
        pred = pred_by_idx.get(record["file_index"])
        if pred is None or pred.get("workflow") is None:
            skipped += 1
            continue
        cases.append(ProcedureCase.build(record, pred))

    print(f"Loaded {len(cases)} cases (skipped {skipped} with no prediction or null workflow)")
    return cases
