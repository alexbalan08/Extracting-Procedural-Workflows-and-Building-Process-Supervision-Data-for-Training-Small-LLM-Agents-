
#for each procedure text we have
#<reasoning> block tracing the logic step by step
#JSON block with the full structured workflow (actions, gateways, execution_states)
#then we have the critic
#structural checker: IDs, reachability, terminal states
#llm checker: missing actions, wrong conditions, incorrect flow
#RAG: always retrieve 1 similar procedure from training set before extraction

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from LLM_Training.inference.checker import StructuralChecker
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from llm_checker import check_with_llm
from retrieval import build_example_pool, retrieve_similar_workflows, format_retrieval_results

#note added to system prompt when RAG is enabled
_RAG_NOTE = (
    "\n\nA similar labelled example from the training set has been provided below your procedure. "
    "Use it as structural reference — pay attention to how gateways, branch conditions, and "
    "execution states are modeled. Do not copy it blindly; adapt to the current procedure."
)


def build_messages(
    procedure_text: str,
    file_index: int | str | None = None,
    use_rag: bool = False,
) -> list[dict]:
    #build full message list: system prompt + 3-shot examples + new procedure
    #RAG note is appended to system prompt when pool is available
    system_content = SYSTEM_PROMPT + (_RAG_NOTE if use_rag else "")
    messages = [{"role": "system", "content": system_content}]

    #important to see the expected output format before the procedure
    for idx, proc, output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Extract the workflow from the following procedure (file_index: {idx}):\n\n{proc}"})
        messages.append({"role": "assistant", "content": output})

    idx_str = f" (file_index: {file_index})" if file_index is not None else ""
    messages.append({"role": "user", "content": f"Extract the workflow from the following procedure{idx_str}:\n\n{procedure_text}"})
    return messages


def parse_response(response_text: str) -> tuple[str, dict | None]:
    #extract reasoning block and JSON workflow from model response
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    json_match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
    if not json_match:
        return reasoning, None
    try:
        workflow = json.loads(json_match.group(1).strip())
    except json.JSONDecodeError:
        return reasoning, None
    return reasoning, workflow


def _run_single_extraction(
    messages: list[dict],
    client: OpenAI,
    model: str,
    pool: list | None,
    embeddings,
    procedure_text: str,
) -> tuple[list[dict], str]:
    #if RAG pool available, always retrieve the most similar procedure before extraction
    #we use the full procedure text as the query — better match than a model-generated description
    if pool is not None:
        results = retrieve_similar_workflows(procedure_text, pool, embeddings, client, k=2)
        retrieved_context = format_retrieval_results(results)
        print(f"  RAG: retrieved example")
        #inject retrieved example as extra context before the model generates
        messages = messages + [{
            "role": "user",
            "content": f"Here is a similar labelled example from the training set for reference:\n\n{retrieved_context}"
        }]

    #single API call — no tool calling needed
    kwargs = dict(model=model, messages=messages, temperature=0.0, max_completion_tokens=8192)
    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": raw})
    return messages, raw


def extract_workflow(
    procedure_text: str,
    client: OpenAI,
    model: str = "gpt-5.4-mini",
    max_attempts: int = 2,
    structural_checker: StructuralChecker | None = None,
    use_llm_checker: bool = True,
    file_index: int | str | None = None,
    pool: list | None = None,
    embeddings=None,
) -> dict:
    #we finally run the orchestration loop

    #call model first and handle rag tool use
    #run structural checker after — if issues feed back and retry
    #for the structural checker go to folder LLM_training and then inference and then checker please!!
    #run LLM semantic checker — if issues feed back and retry
    #If both pass return result

    use_rag = pool is not None
    messages = build_messages(procedure_text, file_index, use_rag)
    issues_feedback = None
    reasoning, workflow = "", None

    for attempt in range(1, max_attempts + 1):
        #we append the feedback from critic to user message
        #so model sees what went wrong
        if issues_feedback:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous extraction had issues. Please fix them and re-extract:\n\n"
                    + "\n".join(f"- {i}" for i in issues_feedback)
                ),
            })

        messages, raw = _run_single_extraction(messages, client, model, pool, embeddings, procedure_text)
        reasoning, workflow = parse_response(raw)

        #if the model output is fucked, ask it to fix the JSON format and retry
        if workflow is None:
            issues_feedback = ["Could not parse JSON from your response. Please output valid JSON in a ```json block."]
            print(f"  Attempt {attempt}: JSON parse failed — retrying...")
            continue

        #structural checks fast rule based
        if structural_checker:
            check_result = structural_checker.check(workflow)
            if not check_result.is_valid:
                issues_feedback = check_result.issues
                print(f"  Attempt {attempt}: {len(check_result.issues)} structural issue(s) — retrying...")
                continue

        #LLM check this takes more and consumes many tokens for input
        if use_llm_checker:
            semantic_issues = check_with_llm(procedure_text, workflow, client, model)
            if semantic_issues:
                issues_feedback = semantic_issues
                print(f"  Attempt {attempt}: {len(semantic_issues)} semantic issue(s) — retrying...")
                continue

        #all checks passed return early with the attempt number 1 or 2 i mean for now
        return {"attempt": attempt, "reasoning": reasoning, "workflow": workflow}

    #max attempts reached
    return {
        "attempt": max_attempts,
        "reasoning": reasoning,
        "workflow": workflow,
        "remaining_issues": issues_feedback,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract workflows with GPT-4o (3-shot + RAG + self-refine)")
    default_input = PROJECT_ROOT / "Data" / "Processed" / "extracted_test.json"
    default_train = PROJECT_ROOT / "Data" / "Processed" / "extracted_train.json"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to input JSON (default: extracted_test.json)")
    parser.add_argument("--train", type=Path, default=None, help="Path to training JSON for RAG pool (optional, default: extracted_train.json)")
    parser.add_argument("--output", type=Path, default=Path("extraction_predictions.json"))
    #parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--max_attempts", type=int, default=3)
    #i keep this for testing
    parser.add_argument("--no_llm_checker", action="store_true", help="Disable LLM semantic checker")
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG retrieval tool")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N procedures")
    parser.add_argument("--limit", type=int, default=10, help="Only process first N procedures")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Export it before running.")
    client = OpenAI(api_key=api_key)
    structural_checker = StructuralChecker()

    #we load the embedded procedures for RAG only once at startup same OpenAI client
    pool, embeddings = None, None
    if not args.no_rag:
        train_path = args.train or default_train
        if train_path.exists():
            print(f"Building RAG pool from {train_path.name} ...")
            pool, embeddings = build_example_pool(train_path, client)
            print(f"  Pool ready: {len(pool)} examples")
        else:
            print(f"Warning: --train path not found ({train_path}), RAG disabled.")

    rag_kwargs = dict(pool=pool, embeddings=embeddings)

    #for batch mode
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}.")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    #for testing one procedure only
    DEBUG_FILE_INDEX = None  #for the file index we want to test

    if DEBUG_FILE_INDEX is not None:
        records = [r for r in records if r.get("file_index") == DEBUG_FILE_INDEX]
        if not records:
            raise ValueError(f"file_index {DEBUG_FILE_INDEX} not found in {args.input}")
    else:
        records = records[args.skip:]
        if args.limit:
            records = records[: args.limit]

    results = []
    for i, record in enumerate(records):
        file_index = record.get("file_index", i)
        procedure_text = record["procedure_text"]
        print(f"[{i+1}/{len(records)}] file_index={file_index}")

        result = extract_workflow(
            procedure_text, client, args.model, args.max_attempts,
            structural_checker, not args.no_llm_checker, file_index,
            **rag_kwargs,
        )
        results.append({
            "file_index": file_index,
            "procedure_text": procedure_text,
            "attempt": result["attempt"],
            "reasoning": result["reasoning"],
            "workflow": result["workflow"],
            "remaining_issues": result.get("remaining_issues"),
        })
        print(f"  → done in {result['attempt']} attempt(s)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
