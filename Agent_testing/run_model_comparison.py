#runs LlamaActionsAgent (method 2 — agent sees the candidate list, picks one) with
#three open-source base models on the held-out set, one at a time. saves per-model
#inference JSONs and prints a comparison table at the end.
#
#models tested here (all 4-bit, fit in 16 GB VRAM via the existing BitsAndBytesConfig):
#  - mistralai/Mistral-7B-Instruct-v0.3
#  - Qwen/Qwen2.5-7B-Instruct
#  - google/gemma-3-12b-it          (note: gated — must accept license on HF Hub)
#
#Llama 3.1 8B and gpt-5.5 are NOT re-run by this script — they're already done.
#Their numbers are shown in the comparison table if their files exist in results/.
#
#  python run_model_comparison.py                  # run all configured models
#  python run_model_comparison.py --model X        # only run one HF model name
#
#skip-existing: if the output file is already there, we don't re-run that model.
#useful for resuming after a crash / GPU OOM / license acceptance hiccup.

import argparse
import contextlib
import gc
import io
import json
import re
from pathlib import Path

from runner import load_cases, run_inference
from agents import LlamaActionsAgent
from evaluate_traces import _compute_metrics


#Qwen 3 hybrid models default to "thinking mode" — their chat template emits a
#<think>...</think> block before the answer. With max_new_tokens=64 the model
#fills the budget with reasoning and never reaches a concrete action name.
#Disabling thinking via the template kwarg gives us short, direct answers
#consistent with how the other base models behave.
class Qwen3ActionsAgent(LlamaActionsAgent):
    def __init__(self, model_name: str, enable_thinking: bool = False,
                 max_new_tokens: int = 64):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        self.enable_thinking = enable_thinking

    def _generate(self, messages: list[dict]) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        #when thinking is on, strip the <think>...</think> block so _best_candidate_match
        #only scans the model's final committed answer. otherwise the matcher might snap
        #to a candidate that the model only MENTIONED inside its reasoning rather than
        #the one it actually concluded with.
        if self.enable_thinking and "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        return raw


#sentinel name we recognise as "Qwen 3 with thinking ON" so it shows as its own
#row in the comparison table without colliding with the thinking-off variant.
#dispatched in _make_agent and re-mapped to the real HF model id at load time.
QWEN3_REASONING_SENTINEL = "Qwen/Qwen3-8B-Reasoning"
QWEN3_REASONING_REAL_NAME = "Qwen/Qwen3-8B"


def _make_agent(model: str) -> LlamaActionsAgent:
    #dispatch on model name. only Qwen 3 needs the chat-template override.
    if model == QWEN3_REASONING_SENTINEL:
        #thinking ON + bigger token budget so the model can reason AND emit the answer
        return Qwen3ActionsAgent(model_name=QWEN3_REASONING_REAL_NAME,
                                  enable_thinking=True, max_new_tokens=8000)
    if "qwen3" in model.lower():
        return Qwen3ActionsAgent(model_name=model, enable_thinking=False)
    return LlamaActionsAgent(model_name=model)


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"

#one place to add / remove models. keep the HF canonical name with the namespace so
#transformers can resolve them directly. all three are chat-template-aware in recent
#transformers, so the existing _LlamaInstructBase._generate works for all of them.
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-NeMo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    QWEN3_REASONING_SENTINEL,        #Qwen 3 8B with thinking mode enabled
    "google/gemma-3-12b-it"
]

#also include these in the comparison table if their files exist — already-done baselines
REFERENCE_FILES = [
    ("llama-3.1-8b-instruct", _HERE / "results" / "inference_llama_actions_predicted.json"),
]


def _short_name(hf_name: str) -> str:
    #drop the org/ namespace and normalise to filename-safe characters
    base = hf_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]", "_", base)


def _output_filename(model: str) -> str:
    return f"inference_{_short_name(model)}_actions_predicted.json"


def run_one(model: str, cases: list) -> bool:
    #returns True if the model's output file exists after this call (either it
    #was already there, or we just produced it). False if the run crashed.
    fn = _output_filename(model)
    path = RESULTS_DIR / fn
    if path.exists():
        print(f"  SKIP — {fn} already exists")
        return True

    print(f"  loading {model}...")
    agent = _make_agent(model)

    try:
        traces = run_inference(cases, agent, give_candidates=True, mode="predicted")
    except Exception as e:
        print(f"  FAILED while running {model}: {e}")
        _free_gpu()
        return False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"  saved {fn}")

    #release GPU memory before the next model loads; bitsandbytes / quantized
    #weights don't always free on GC alone
    del agent
    _free_gpu()
    return True


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_actions_lookup() -> tuple[list, dict]:
    #load_cases prints a "Loaded N cases" line we don't need; swallow it
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_cases(DEFAULT_HELD_OUT, DEFAULT_PREDICTIONS)
    actions_lookup: dict = {}
    for c in cases:
        actions_lookup[(c.file_index, "predicted")] = c.pred_action_names
        actions_lookup[(c.file_index, "gold")] = c.gold_action_names
    return cases, actions_lookup


def _row_for(name: str, path: Path, actions_lookup: dict) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        traces = json.load(f)
    if not traces:
        return None
    return {"model": name, **_compute_metrics(traces, actions_lookup)}


def print_comparison() -> None:
    print("\n\n=== Comparison (method 2: LlamaActionsAgent, --graph predicted) ===")
    _, actions_lookup = _load_actions_lookup()

    rows: list[dict] = []
    for m in MODELS:
        row = _row_for(_short_name(m), RESULTS_DIR / _output_filename(m), actions_lookup)
        if row:
            rows.append(row)
    for name, path in REFERENCE_FILES:
        row = _row_for(name, path, actions_lookup)
        if row:
            rows.append(row)

    if not rows:
        print("(no result files yet)")
        return

    keys = ["model", "procedures",
            "step_valid_%", "first_step_valid_%", "completed_%", "invented_%"]
    widths = [max(len(k), max(len(_fmt(r.get(k, ""))) for r in rows)) for k in keys]

    line = " | ".join(k.ljust(w) for k, w in zip(keys, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(_fmt(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LlamaActionsAgent across base models on the held-out set."
    )
    parser.add_argument("--model", default=None,
                        help="Single HF model name to run. If omitted, runs all configured.")
    args = parser.parse_args()

    cases, _ = _load_actions_lookup()
    models = [args.model] if args.model else MODELS

    for m in models:
        print(f"\n=== {m} ===")
        run_one(m, cases)

    print_comparison()


if __name__ == "__main__":
    main()
