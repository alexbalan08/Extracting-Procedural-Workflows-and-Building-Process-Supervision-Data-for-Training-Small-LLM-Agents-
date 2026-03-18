"""
GPT-4o workflow extraction with 3-shot learning, on-demand RAG, and self-refine loop.

For each procedure text, the model outputs:
  A <reasoning> block tracing the logic step by step
  A JSON block with the full structured workflow (actions, gateways, execution_states)

Checking pipeline per attempt:
  1. Structural checker (rule-based): IDs, reachability, terminal states
  2. LLM checker (semantic): missing actions, wrong conditions, incorrect flow

RAG: the model can call retrieve_similar_workflows() via OpenAI tool calling
when uncertain about gateway type or flow structure.
"""

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

# ── RAG tool note appended to SYSTEM_PROMPT when a pool is loaded ─────────────
_RAG_NOTE = (
    "\n\nYou have access to a retrieval tool. Use it ONLY as a last resort — when you "
    "are genuinely unable to determine a gateway type or branching structure from the "
    "procedure text alone. Do NOT call it for straightforward procedures or simple flows. "
    "If you can reason through the structure confidently, extract directly without retrieval."
)

# ── OpenAI tool schema ─────────────────────────────────────────────────────────
_RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_similar_workflows",
        "description": (
            "Retrieve the single most similar labelled workflow example from the training set. "
            "Call ONLY when you cannot confidently determine gateway type or branching structure "
            "from the procedure text alone. Do not call for simple or clear procedures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short description of the workflow pattern you are uncertain about.",
                },
            },
            "required": ["query"],
        },
    },
}


def build_messages(
    procedure_text: str,
    file_index: int | str | None = None,
    use_rag: bool = False,
) -> list[dict]:
    """Build the full message list: system prompt + 3-shot examples + new procedure."""
    system_content = SYSTEM_PROMPT + (_RAG_NOTE if use_rag else "")
    messages = [{"role": "system", "content": system_content}]
    for idx, proc, output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Extract the workflow from the following procedure (file_index: {idx}):\n\n{proc}"})
        messages.append({"role": "assistant", "content": output})
    idx_str = f" (file_index: {file_index})" if file_index is not None else ""
    messages.append({"role": "user", "content": f"Extract the workflow from the following procedure{idx_str}:\n\n{procedure_text}"})
    return messages


def parse_response(response_text: str) -> tuple[str, dict | None]:
    """Extract reasoning block and JSON workflow from model response."""
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
    embedder,
) -> tuple[list[dict], str]:
    """Inner loop: call the model and handle any tool calls before returning the final raw response.

    Handles up to 3 tool-call rounds. Returns the updated messages list and the
    final raw text response (ready for parse_response).
    """
    use_rag = pool is not None
    tools = [_RETRIEVAL_TOOL] if use_rag else None

    for _ in range(3):
        kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=4096)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # append assistant message with tool_calls
            messages.append(choice.message)

            # process every tool call in this round
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args["query"]

                results = retrieve_similar_workflows(query, pool, embeddings, embedder, k=1)
                tool_result = format_retrieval_results(results)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
            # continue inner loop — model will now produce a real response
            continue

        # normal response — return raw text
        raw = choice.message.content
        messages.append({"role": "assistant", "content": raw})
        return messages, raw

    # exhausted tool-call rounds — return last content (may be None)
    raw = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": raw})
    return messages, raw


def extract_workflow(
    procedure_text: str,
    client: OpenAI,
    model: str = "gpt-4o",
    max_attempts: int = 2,
    structural_checker: StructuralChecker | None = None,
    use_llm_checker: bool = True,
    file_index: int | str | None = None,
    pool: list | None = None,
    embeddings=None,
    embedder=None,
) -> dict:
    """Run the extraction + self-refine loop (up to max_attempts).

    Each attempt:
      1. Call model (inner loop handles RAG tool calls transparently)
      2. Run structural checker — if issues, feed back and retry
      3. Run LLM semantic checker — if issues, feed back and retry
      4. If both pass, return result
    """
    use_rag = pool is not None
    messages = build_messages(procedure_text, file_index, use_rag)
    issues_feedback = None
    reasoning, workflow = "", None

    for attempt in range(1, max_attempts + 1):
        # append feedback from previous attempt if any
        if issues_feedback:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous extraction had issues. Please fix them and re-extract:\n\n"
                    + "\n".join(f"- {i}" for i in issues_feedback)
                ),
            })

        # extract (inner loop handles tool calls)
        messages, raw = _run_single_extraction(messages, client, model, pool, embeddings, embedder)
        reasoning, workflow = parse_response(raw)

        if workflow is None:
            issues_feedback = ["Could not parse JSON from your response. Please output valid JSON in a ```json block."]
            print(f"  Attempt {attempt}: JSON parse failed — retrying...")
            continue

        # step 1: structural checks (fast, rule-based)
        if structural_checker:
            structural_issues = structural_checker.check(workflow)
            if structural_issues:
                issues_feedback = structural_issues
                print(f"  Attempt {attempt}: {len(structural_issues)} structural issue(s) — retrying...")
                continue

        # step 2: LLM semantic checks
        if use_llm_checker:
            semantic_issues = check_with_llm(procedure_text, workflow, client, model)
            if semantic_issues:
                issues_feedback = semantic_issues
                print(f"  Attempt {attempt}: {len(semantic_issues)} semantic issue(s) — retrying...")
                continue

        # both checks passed
        return {"attempt": attempt, "reasoning": reasoning, "workflow": workflow}

    # return best effort on final attempt
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
    parser.add_argument("--text", type=str, help="Single procedure text (instead of --input)")
    parser.add_argument("--file_index", type=str, default=None, help="file_index for --text mode")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--no_llm_checker", action="store_true", help="Disable LLM semantic checker")
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG retrieval tool")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N procedures")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N procedures")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Export it before running.")
    client = OpenAI(api_key=api_key)
    structural_checker = StructuralChecker()

    # load RAG pool once at startup
    pool, embeddings, embedder = None, None, None
    if not args.no_rag:
        train_path = args.train or default_train
        if train_path.exists():
            print(f"Building RAG pool from {train_path.name} ...")
            pool, embeddings, embedder = build_example_pool(train_path)
            print(f"  Pool ready: {len(pool)} examples")
        else:
            print(f"Warning: --train path not found ({train_path}), RAG disabled.")

    rag_kwargs = dict(pool=pool, embeddings=embeddings, embedder=embedder)

    # single text mode
    if args.text:
        result = extract_workflow(
            args.text, client, args.model, args.max_attempts,
            structural_checker, not args.no_llm_checker, args.file_index,
            **rag_kwargs,
        )
        print("\n─── REASONING ───────────────────────────────────────────")
        print(result["reasoning"])
        print("\n─── WORKFLOW JSON ───────────────────────────────────────")
        print(json.dumps(result["workflow"], indent=2, ensure_ascii=False))
        return

    # batch mode
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}.")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

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
