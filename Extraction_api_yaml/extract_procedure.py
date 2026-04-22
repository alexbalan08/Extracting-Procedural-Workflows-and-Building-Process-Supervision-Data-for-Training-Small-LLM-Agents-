# Extraction pipeline: procedure text → structured workflow YAML + execution states
# Same as JSON version but prompts and parsing use YAML format
# Output file is still JSON for pipeline compatibility with validation and PRM scripts

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from openai import OpenAI, BadRequestError

PROJECT_ROOT = Path(__file__).parent.parent

from structural_checker import StructuralChecker
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from llm_checker import check_with_llm, ReflexionMemory
from retrieval import build_example_pool, retrieve_similar_workflows, format_retrieval_results
from trace_builder import build_traces

#note added to system prompt when RAG is enabled
_RAG_NOTE = (
    "\n\nA similar labelled example from the training set has been provided below your procedure. "
    "Use it as structural reference — pay attention to how gateways, branch conditions, and "
    "flow edges are modeled. Do not copy it blindly; adapt to the current procedure."
)


def _sanitize(text: str) -> str:
    # remove control characters, null bytes, and lone surrogates (U+D800-DFFF)
    # lone surrogates are valid Python str chars but cause json.dumps to fail
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def build_messages(
    procedure_text: str,
    file_index: int | str | None = None,
    use_rag: bool = False,
) -> list[dict]:
    #build full message list: system prompt + 3-shot examples + new procedure
    #RAG note is appended to system prompt when pool is available
    procedure_text = _sanitize(procedure_text)
    system_content = SYSTEM_PROMPT + (_RAG_NOTE if use_rag else "")
    messages = [{"role": "system", "content": system_content}]

    #important to see the expected output format before the procedure
    for idx, proc, output in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": f"Extract the workflow from the following procedure (file_index: {idx}):\n\n{proc}",
            }
        )
        messages.append({"role": "assistant", "content": output})

    idx_str = f" (file_index: {file_index})" if file_index is not None else ""
    messages.append(
        {
            "role": "user",
            "content": f"Extract the workflow from the following procedure{idx_str}:\n\n{procedure_text}",
        }
    )
    return messages


def parse_response(response_text: str) -> tuple[str, dict | None]:
    #extract reasoning block and YAML workflow from model response
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    #try yaml block first then fall back to json block in case the model ignores the format instruction
    yaml_match = re.search(r"```ya?ml\s*(.*?)```", response_text, re.DOTALL)
    json_match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)

    block = yaml_match or json_match
    if not block:
        return reasoning, None
    try:
        workflow = yaml.safe_load(block.group(1).strip())
    except yaml.YAMLError:
        #fallback to json.loads in case it was actually json inside yaml block
        try:
            workflow = json.loads(block.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            return reasoning, None
    return reasoning, workflow


def _run_single_extraction(
    messages: list[dict],
    client: OpenAI,
    model: str,
) -> tuple[list[dict], str]:
    #single API call — no tool calling needed
    # sanitize all message content to strip control characters that cause 400 errors
    clean_messages = [
        {**m, "content": _sanitize(m["content"]) if isinstance(m.get("content"), str) else m.get("content")}
        for m in messages
    ]
    kwargs = dict(model=model, messages=clean_messages, temperature=0.0, max_completion_tokens=8192, seed=42)
    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    usage = response.usage
    messages.append({"role": "assistant", "content": raw})
    return messages, raw, usage


def extract_workflow(
    procedure_text: str,
    client: OpenAI,
    model: str = "gpt-5.4",
    max_attempts: int = 2,
    structural_checker: StructuralChecker | None = None,
    use_llm_checker: bool = True,
    file_index: int | str | None = None,
    pool: list | None = None,
    embeddings=None,
    memory: ReflexionMemory | None = None,
    rag_k: int = 2,
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
    total_completion_tokens = 0

    #retrieve once per procedure and reuse the same context across retries
    if pool is not None:
        results = retrieve_similar_workflows(procedure_text, pool, embeddings, client, k=rag_k)
        retrieved_context = format_retrieval_results(results)
        print(f"  RAG: retrieved {len(results)} example(s)")
        messages.append(
            {
                "role": "user",
                "content": f"Here is a similar labelled example from the training set for reference:\n\n{retrieved_context}",
            }
        )

    for attempt in range(1, max_attempts + 1):
        #we append the feedback from critic to user message
        #so model sees what went wrong
        if issues_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous extraction had issues. Please fix them and re-extract:\n\n"
                        + "\n".join(f"- {i}" for i in issues_feedback)
                    ),
                }
            )

        messages, raw, usage = _run_single_extraction(messages, client, model)
        if usage:
            total_completion_tokens += usage.completion_tokens
        reasoning, workflow = parse_response(raw)

        #if the model output is fucked, ask it to fix the YAML format and retry
        if workflow is None:
            issues_feedback = ["Could not parse YAML from your response. Please output valid YAML in a ```yaml block."]
            print(f"  Attempt {attempt}: YAML parse failed — retrying...")
            continue

        #structural checks fast rule based
        if structural_checker:
            check_result = structural_checker.check(workflow)
            if not check_result.is_valid:
                issues_feedback = check_result.issues
                print(f"  Attempt {attempt}: {len(check_result.issues)} structural issue(s) — retrying...")
                continue

        if use_llm_checker:
            semantic_issues = check_with_llm(procedure_text, workflow, client, model,
                                                memory=memory, file_index=file_index, fmt="yaml")
            if semantic_issues:
                issues_feedback = semantic_issues
                print(f"  Attempt {attempt}: {len(semantic_issues)} semantic issue(s) — retrying...")
                continue

        #all checks passed return early with the attempt number 3 i mean for now
        return {"attempt": attempt, "reasoning": reasoning, "workflow": workflow,
                "completion_tokens": total_completion_tokens}

    #max attempts reached
    return {
        "attempt": max_attempts,
        "reasoning": reasoning,
        "workflow": workflow,
        "remaining_issues": issues_feedback,
        "completion_tokens": total_completion_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract workflows with GPT 5.4 mini — YAML format (3-shot + RAG + self-refine)")
    default_input = PROJECT_ROOT / "Data" / "Processed" / "extracted_test.json"
    default_train = PROJECT_ROOT / "Data" / "Processed" / "extracted_train.json"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to input JSON (default: extracted_test.json)")
    parser.add_argument("--train", type=Path, default=None, help="Path to training JSON for RAG pool (optional, default: extracted_train.json)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Extraction_results" / "extraction_predictions_yaml.yaml")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG retrieval tool")
    parser.add_argument("--rag_k", type=int, default=1, help="Number of similar procedures to retrieve")
    parser.add_argument("--no_checker", action="store_true", help="Disable LLM semantic checker (critic)")
    parser.add_argument("--no_memory", action="store_true", help="Disable reflexion memory across procedures")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N procedures")
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

    rag_kwargs = dict(pool=pool, embeddings=embeddings, rag_k=args.rag_k)

    #reflexion memory persists across procedures so the checker learns from past mistakes
    reflexion_memory = None if args.no_memory else ReflexionMemory()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}.")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    if args.limit:
        records = records[: args.limit]

    # resume: load already-processed results so a crashed run can continue
    results = []
    done_indices = set()
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            results = yaml.safe_load(f) or []
        done_indices = {r["file_index"] for r in results}
        if done_indices:
            print(f"Resuming — {len(done_indices)} already done: {sorted(done_indices)}")

    for i, record in enumerate(records):
        file_index = record.get("file_index", i)
        if file_index in done_indices:
            print(f"[{i+1}/{len(records)}] file_index={file_index}  (skipped — already done)")
            continue

        procedure_text = record["procedure_text"]
        print(f"[{i+1}/{len(records)}] file_index={file_index}")

        try:
            result = extract_workflow(
                procedure_text,
                client,
                args.model,
                args.max_attempts,
                structural_checker,
                not args.no_checker,
                file_index,
                **rag_kwargs,
                memory=reflexion_memory,
            )
        except BadRequestError as e:
            print(f"  WARNING: skipping file_index={file_index} — BadRequestError: {e}")
            continue
        execution_states = build_traces(result["workflow"])
        results.append(
            {
                "file_index": file_index,
                "procedure_text": procedure_text,
                "attempt": result["attempt"],
                "reasoning": result["reasoning"],
                "workflow": result["workflow"],
                "execution_states": execution_states,
                "remaining_issues": result.get("remaining_issues"),
                "completion_tokens": result.get("completion_tokens", 0),
            }
        )
        print(f"  -> done in {result['attempt']} attempt(s), {len(execution_states)} execution states, "
              f"{result.get('completion_tokens', 0)} output tokens")

        # save after every record so a crash doesn't lose progress
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(results, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    #completion tokens per procedure — the key metric for json vs yaml format efficiency
    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    print(f"\nSaved {len(results)} results to {args.output}")
    if results:
        print(f"Avg completion tokens per procedure: {total_completion // len(results):,}")


if __name__ == "__main__":
    main()
