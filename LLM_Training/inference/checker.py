"""Workflow checkers — StructuralChecker (rule-based) + LLMChecker (same model, critic role).
Use CombinedChecker in the agent."""

import json
import re
from dataclasses import dataclass, field

from utils import generate


# ── Checker prompt ────────────────────────────────────────────────────────────

CHECKER_SYSTEM_PROMPT = """You are a workflow extraction verifier. Your job is to review an extracted workflow against the original procedure text and decide whether the extraction is correct.

Check all of the following:
1. **Completeness** – every action/step described in the text is represented as an action in the workflow.
2. **No hallucinations** – no actions exist in the workflow that are not described in the text.
3. **Flow correctness** – predecessor/successor connections match the sequential or conditional flow described.
4. **Gateways** – decision points ("if", "either…or", "otherwise") and parallel splits ("at the same time", "simultaneously") are captured with the correct type (exclusive/parallel/inclusive) and role (split/merge).
5. **Actors** – each action is assigned to the correct actor mentioned in the text (or null if unspecified).
6. **Execution states** – the listed states correctly reflect reachable snapshots of workflow progress.

Respond with EXACTLY one of these two formats:

If the extraction is correct:
<verdict>VALID</verdict>

If the extraction has problems:
<verdict>INVALID</verdict>
<issues>
- <specific issue 1>
- <specific issue 2>
...
</issues>

Be concrete: name the specific action/gateway/field that is wrong and what the correct value should be."""


def _format_checker_message(procedure_text: str, workflow: dict) -> str:
    workflow_str = json.dumps(workflow, indent=2, ensure_ascii=False)
    return (
        f"Procedure text:\n\"\"\"\n{procedure_text}\n\"\"\"\n\n"
        f"Extracted workflow:\n```json\n{workflow_str}\n```\n\n"
        "Is this extraction correct? Review it carefully against the procedure text."
    )


# ── CheckResult ───────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    is_valid: bool
    issues: list = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "Extraction is valid."
        return (
            f"Found {len(self.issues)} issue(s):\n"
            + "\n".join(f"  {i+1}. {iss}" for i, iss in enumerate(self.issues))
        )


# ── StructuralChecker ─────────────────────────────────────────────────────────

class StructuralChecker:
    """Rule-based pre-flight: unique IDs, start reachability, terminal state."""

    @staticmethod
    def _check_unique_ids(items, label):
        seen, dupes = set(), set()
        for item in items:
            val = item.get("id", "")
            (dupes if val in seen else seen).add(val)
        return [f"Duplicate {label} ID: '{d}'." for d in sorted(dupes)]

    @staticmethod
    def _check_start_reachability(w):
        """Gateway-aware reachability from 'start'."""
        actions = w.get("actions", [])
        gateways = w.get("gateways", [])
        if not actions:
            return []

        action_ids = {a["id"] for a in actions}
        # gateway → list of next IDs it can reach
        gw_targets = {
            g["id"]: [b["next"] for b in g.get("branches", []) if b.get("next")]
            for g in gateways
        }

        reachable = set()
        visited_gw = set()
        # seed: actions/gateways directly following "start"
        frontier = set()
        for a in actions:
            if "start" in a.get("predecessors", []):
                reachable.add(a["id"])
                frontier.update(a.get("successors", []))
        for g in gateways:
            if "start" in g.get("incoming_from", []):
                frontier.add(g["id"])

        changed = True
        while changed:
            changed = False
            new_frontier = set()
            for nid in frontier:
                if nid in gw_targets and nid not in visited_gw:
                    visited_gw.add(nid)
                    new_frontier.update(gw_targets[nid])
                    changed = True
                elif nid in action_ids and nid not in reachable:
                    reachable.add(nid)
                    a = next(a for a in actions if a["id"] == nid)
                    new_frontier.update(a.get("successors", []))
                    changed = True
            # also check actions whose predecessors are now reachable
            for a in actions:
                if a["id"] not in reachable:
                    for p in a.get("predecessors", []):
                        if p in reachable or p in visited_gw:
                            reachable.add(a["id"])
                            new_frontier.update(a.get("successors", []))
                            changed = True
                            break
            frontier = new_frontier

        return [
            f"Action '{a['id']}' is not reachable from 'start'."
            for a in actions if a["id"] not in reachable
        ]

    @staticmethod
    def _check_terminal_state(w):
        states = w.get("execution_states", [])
        if not states:
            return []
        if not any(s.get("can_terminate") or not s.get("available_next") for s in states):
            return ["No terminal execution state found."]
        return []

    def check(self, workflow) -> CheckResult:
        if workflow is None:
            return CheckResult(False, ["Output could not be parsed as valid JSON."])

        issues = self._check_unique_ids(workflow.get("actions", []), "action")
        issues += self._check_unique_ids(workflow.get("gateways", []), "gateway")
        issues += self._check_start_reachability(workflow)
        issues += self._check_terminal_state(workflow)
        return CheckResult(len(issues) == 0, issues)


# ── LLMChecker ────────────────────────────────────────────────────────────────

class LLMChecker:
    """Same model as extractor, different system prompt — acts as a semantic critic."""

    def __init__(self, model, tokenizer, max_new_tokens=512, temperature=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @staticmethod
    def _parse_response(response: str):
        verdict_match = re.search(r"<verdict>(.*?)</verdict>", response, re.IGNORECASE | re.DOTALL)
        verdict = verdict_match.group(1).strip().upper() if verdict_match else ""
        is_valid = verdict == "VALID"

        issues = []
        if not is_valid:
            issues_match = re.search(r"<issues>(.*?)</issues>", response, re.IGNORECASE | re.DOTALL)
            if issues_match:
                for line in issues_match.group(1).splitlines():
                    line = re.sub(r"^[-*•]\s*", "", line).strip()
                    if line:
                        issues.append(line)
            if not issues:
                issues = [f"LLM checker flagged extraction as invalid: {response.strip()[:400]}"]
        return is_valid, issues

    def check(self, workflow: dict, procedure_text: str) -> CheckResult:
        messages = [
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": _format_checker_message(procedure_text, workflow)},
        ]
        response = generate(self.model, self.tokenizer, messages,
                            self.max_new_tokens, self.temperature)
        is_valid, issues = self._parse_response(response)
        return CheckResult(is_valid, issues)


# ── CombinedChecker ───────────────────────────────────────────────────────────

class CombinedChecker:
    """Stage 1: structural pre-flight (instant). Stage 2: LLM semantic check (only if stage 1 passes)."""

    def __init__(self, model, tokenizer, **llm_kwargs):
        self.structural = StructuralChecker()
        self.llm = LLMChecker(model, tokenizer, **llm_kwargs)

    def check(self, workflow, procedure_text: str) -> CheckResult:
        result = self.structural.check(workflow)
        if not result.is_valid:
            return result
        return self.llm.check(workflow, procedure_text)
