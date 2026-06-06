"""Step 2 wrapper: run the real extraction pipeline on procedure text.

Thin adapter around Extraction_api/extract_procedure.extract_workflow so the
Streamlit app never has to know about the project's sys.path quirks. Also exposes
the saved predictions / held-out gold so the demo can run fully offline when no
API key is available.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTION_DIR = PROJECT_ROOT / "Extraction_api"
DATA = PROJECT_ROOT / "Data" / "Processed"
HELD_OUT_PATH = PROJECT_ROOT / "Agent_testing" / "held_out.json"

# Extraction_api uses bare imports (from structural_checker import ...) that assume
# its own folder is importable. No module names collide with Agent_testing (verified),
# so adding it to sys.path is safe.
if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))


def _extractor():
    # imported lazily so the app starts even if openai isn't configured yet
    import extract_procedure as ep
    from structural_checker import StructuralChecker
    from llm_checker import ReflexionMemory
    return ep, StructuralChecker, ReflexionMemory


def make_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def run_extraction(
    procedure_text: str,
    client,
    model: str = "gpt-5.4-mini",
    max_attempts: int = 3,
    use_llm_checker: bool = True,
    use_structural_checker: bool = True,
    file_index: int | str | None = None,
) -> dict:
    """Live extraction: returns {workflow, reasoning, attempt, execution_states,
    remaining_issues, completion_tokens}. RAG is left off for demo speed/cost."""
    ep, StructuralChecker, ReflexionMemory = _extractor()
    checker = StructuralChecker() if use_structural_checker else None
    result = ep.extract_workflow(
        procedure_text,
        client,
        model=model,
        max_attempts=max_attempts,
        structural_checker=checker,
        use_llm_checker=use_llm_checker,
        file_index=file_index,
        memory=ReflexionMemory() if use_llm_checker else None,
    )
    from trace_builder import build_traces
    result["execution_states"] = build_traces(result.get("workflow"))
    return result


# --------------------------------------------------------------- offline data
@lru_cache(maxsize=1)
def _held_out() -> dict:
    if not HELD_OUT_PATH.exists():
        return {}
    with open(HELD_OUT_PATH, encoding="utf-8") as f:
        return {r["file_index"]: r for r in json.load(f)}


def held_out_index() -> list[dict]:
    """List of held-out procedures (those we have gold for), shortest first —
    these are the ones that can be graded green/red in Step 5."""
    out = []
    for fi, rec in _held_out().items():
        out.append({
            "file_index": fi,
            "n_chars": len(rec.get("procedure_text", "")),
            "n_actions": len((rec.get("workflow") or {}).get("actions") or []),
            "preview": (rec.get("procedure_text", "")[:80] + "…"),
        })
    return sorted(out, key=lambda r: r["n_chars"])



def held_out_record(file_index) -> dict | None:
    return _held_out().get(file_index)
