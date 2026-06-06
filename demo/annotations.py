"""Human-in-the-loop annotation store (the data flywheel).

Each save appends one JSON object as a line to demo/annotations/labeled_data.jsonl.
Two record kinds:
  * "extraction_labels" — per-action / per-gateway correct/wrong on an extracted graph
    (produces gold-style labeled documents)
  * "rollout_labels"    — per-step correct/wrong on an agent rollout
    (produces step-level process-supervision labels for PRM / agent retraining)

JSONL keeps it append-only and trivially re-readable by the training pipelines later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ANNOT_DIR = Path(__file__).resolve().parent / "annotations"
ANNOT_PATH = ANNOT_DIR / "labeled_data.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_record(record: dict) -> dict:
    """Stamp with a UTC timestamp and append one line. Returns the stored record."""
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _now(), **record}
    with open(ANNOT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_all() -> list[dict]:
    if not ANNOT_PATH.exists():
        return []
    out = []
    with open(ANNOT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def summary() -> dict:
    recs = load_all()
    n_ext = sum(1 for r in recs if r.get("kind") == "extraction_labels")
    n_roll = sum(1 for r in recs if r.get("kind") == "rollout_labels")
    # count individual labeled units, not just records
    n_action_labels = sum(len(r.get("action_labels") or {})
                          for r in recs if r.get("kind") == "extraction_labels")
    n_step_labels = sum(len(r.get("step_labels") or [])
                        for r in recs if r.get("kind") == "rollout_labels")
    return {
        "records": len(recs),
        "extraction_records": n_ext,
        "rollout_records": n_roll,
        "action_labels": n_action_labels,
        "step_labels": n_step_labels,
    }


def raw_bytes() -> bytes:
    """The whole JSONL file as bytes, for a download button."""
    if not ANNOT_PATH.exists():
        return b""
    return ANNOT_PATH.read_bytes()
