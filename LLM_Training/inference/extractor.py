"""Workflow extractor: loads the fine-tuned model and generates structured
workflow JSON + reasoning trace from procedure text."""

import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))
from system_prompt import EXTRACTION_SYSTEM_PROMPT
from feedback_prompt import format_initial_user_message, format_feedback_user_message
from utils import generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def parse_response(response: str) -> dict:
    """Extract reasoning and workflow JSON from model output."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    json_match = re.search(r"```json\s*([\s\S]+?)```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        start = response.find("{")
        if start != -1:
            depth, end = 0, -1
            for i, ch in enumerate(response[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = response[start:end] if end != -1 else ""
        else:
            json_str = ""

    try:
        workflow = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        workflow = None

    return {"reasoning": reasoning, "workflow": workflow, "raw": response}


class WorkflowExtractor:
    """Thin wrapper around a causal LM that extracts structured workflows."""

    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, load_in_4bit=True, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_messages(self, procedure_text, feedback_issues=None, attempt=1):
        if feedback_issues:
            user_content = format_feedback_user_message(procedure_text, feedback_issues, attempt)
        else:
            user_content = format_initial_user_message(procedure_text)
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def extract(self, procedure_text, feedback_issues=None, attempt=1):
        messages = self._build_messages(procedure_text, feedback_issues, attempt)
        raw = generate(self.model, self.tokenizer, messages, self.max_new_tokens, self.temperature)
        return parse_response(raw)
