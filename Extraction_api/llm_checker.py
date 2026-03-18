"""
LLM-based semantic checker for extracted workflows.

Uses GPT-4o to verify the extracted workflow against the original procedure text,
catching issues the structural checker misses (missing actions, wrong conditions, etc.).
Returns a list of specific issues for the self-refine feedback loop.
"""

from openai import OpenAI

CHECKER_SYSTEM_PROMPT = """You are an expert workflow validator. You will be given:
1. An original procedure text
2. An extracted workflow JSON

Your job is to carefully compare the two and identify REAL errors only.

Check for:
- Missing actions: steps mentioned in the text that are not in the workflow
- Extra actions: actions in the workflow not supported by the text
- Wrong action names: names that paraphrase instead of using exact wording from the text
- Missing gateways: branching/conditions in the text not modeled as gateways
- Wrong gateway type: e.g. exclusive used when the text implies parallel or inclusive
- Wrong branch conditions: condition labels that don't match the text
- Wrong flow: predecessor/successor relationships that contradict the text order
- Execution states: obvious missing paths (e.g. a branch exists in gateways but has no corresponding states)

Output ONLY a JSON array of issue strings. Each issue must be specific and actionable.
If no issues are found, output an empty array: []

Example output:
[
  "Action 'Review Request' is mentioned in the text but missing from the workflow",
  "Gateway gateway_xor_2 has condition 'Approved' but the text says 'Accepted'",
  "Action 'send_notification' should come after 'validate_form' according to the text, but predecessors show 'start'"
]

Do NOT flag issues with execution_states coverage or ID formatting — those are handled separately.
Do NOT invent issues. Only flag clear, verifiable problems."""


def check_with_llm(
    procedure_text: str,
    workflow: dict,
    client: OpenAI,
    model: str = "gpt-4o",
) -> list[str]:
    """Run the LLM checker and return a list of issues (empty = no issues)."""
    import json

    user_message = f"""PROCEDURE TEXT:
{procedure_text}

EXTRACTED WORKFLOW:
{json.dumps(workflow, indent=2, ensure_ascii=False)}

Identify any issues with the extracted workflow compared to the procedure text."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        # handle both {"issues": [...]} and bare [...]
        if isinstance(parsed, list):
            return parsed
        for key in ("issues", "errors", "problems"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return []
    except (json.JSONDecodeError, KeyError):
        return []
