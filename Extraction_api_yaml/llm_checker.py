



#use same model as critic to verify the extracted workflow against the original procedure text,
#the structural checker catches structure issues (missing actions, wrong conditions, etc.)
#return the feedback for next iteration
#YAML version: shows workflow as YAML to the checker model
#
#reflexion memory: the checker remembers issues found in previous procedures
#so it learns from past mistakes and catches recurring patterns

from openai import OpenAI
import yaml

#max number of past issues to keep in memory so we dont blow up the context window
MAX_MEMORY_SIZE = 15


class ReflexionMemory:
    #keeps a running list of issues found across procedures
    #the checker sees these as context so it can catch recurring patterns
    #same class as in the json checker but kept local to avoid sys.path hacks

    def __init__(self, max_size=MAX_MEMORY_SIZE):
        self.issues = []
        self.max_size = max_size

    def add(self, new_issues, file_index=None):
        for issue in new_issues:
            entry = f"[file_index={file_index}] {issue}" if file_index else issue
            self.issues.append(entry)
        if len(self.issues) > self.max_size:
            self.issues = self.issues[-self.max_size:]

    def format_for_prompt(self, fmt="yaml"):
        if not self.issues:
            return ""
        header = (
            "\n\n─── REFLEXION MEMORY ──────────────────────────────────────────────────────\n"
            "Below are issues found in PREVIOUS extractions. Use them to catch recurring patterns.\n"
            "Do NOT blindly copy these — only flag an issue if it actually exists in the current extraction.\n\n"
        )
        if fmt == "yaml":
            body = yaml.dump(self.issues, allow_unicode=True, default_flow_style=False)
        else:
            import json
            body = json.dumps(self.issues, indent=2, ensure_ascii=False)
        return header + body

    def __len__(self):
        return len(self.issues)

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
    memory: ReflexionMemory | None = None,
    file_index: int | str | None = None,
    fmt: str = "yaml",
) -> list[str]:
    """Run the LLM checker and return a list of issues (empty = no issues)."""
    import json

    #build the system prompt with reflexion memory if available
    system_prompt = CHECKER_SYSTEM_PROMPT
    if memory and len(memory) > 0:
        system_prompt += memory.format_for_prompt(fmt=fmt)

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
            {"role": "system", "content": system_prompt},
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
            issues = parsed
        else:
            issues = []
            for key in ("issues", "errors", "problems"):
                if key in parsed and isinstance(parsed[key], list):
                    issues = parsed[key]
                    break

        #add found issues to reflexion memory for future procedures
        if memory and issues:
            memory.add(issues, file_index=file_index)

        return issues
    except (json.JSONDecodeError, KeyError):
        #skip the check if parsing fails
        return []
