



#use same model as critic to verify the extracted workflow against the original procedure text,
#the structural checker catches structure issues (missing actions, wrong conditions, etc.)
#return the feedback for hext iteration
#YAML version: shows workflow as YAML to the checker model

from openai import OpenAI
import yaml

CHECKER_SYSTEM_PROMPT = """You are an expert workflow validator. You will be given:
1. An original procedure text
2. An extracted workflow in YAML format

Your job is to identify REAL structural errors only. Be conservative — only flag issues you are certain about.

Check for:
- Missing actions: a clearly named activity in the text is completely absent from the workflow
- Extra actions: an action in the workflow has no support in the text whatsoever
- Wrong gateway type: exclusive used when text clearly implies parallel or inclusive (or vice versa)
- Wrong branch conditions: condition labels that clearly contradict the text
- Wrong flow: predecessor/successor order that directly contradicts the text sequence

Do NOT flag:
- Missing data objects, forms, or documents (e.g. "SAP system", "invoice form") — these are not actions
- Explicit end/termination nodes — empty successors lists are a valid way to model process end
- "request is handled" or similar state descriptions — these are states, not actions
- Ambiguous loops or repeated checks where the text is unclear
- Minor naming differences (gerund vs imperative form, pronoun vs full name)
- Missing actor fields — actors are not part of the extraction schema
- Execution states coverage or ID formatting — handled separately

Output ONLY a JSON array of issue strings. Each issue must be specific and actionable.
If no issues are found, output an empty array: []

Example output:
[
  "Action 'Review Request' is mentioned in the text but missing from the workflow",
  "Gateway gateway_xor_2 has condition 'Approved' but the text says 'Accepted'"
]"""


def check_with_llm(
    procedure_text: str,
    workflow: dict,
    client: OpenAI,
    model: str = "gpt-4o",
) -> list[str]:
    """Run the LLM checker and return a list of issues (empty = no issues)."""
    import json

    #show workflow as YAML to the checker for consistency with the extraction format
    workflow_yaml = yaml.dump(workflow, allow_unicode=True, sort_keys=False, default_flow_style=False)

    user_message = f"""PROCEDURE TEXT:
{procedure_text}

EXTRACTED WORKFLOW:
{workflow_yaml}

Identify any issues with the extracted workflow compared to the procedure text."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_completion_tokens=1024,
        seed=42,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        #ideal case: model returns array ["issue1", "issue2"]
        if isinstance(parsed, list):
            return parsed

        for key in ("issues", "errors", "problems"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]

        return []
    except (json.JSONDecodeError, KeyError):
        #skip the check if parsing fails
        return []
