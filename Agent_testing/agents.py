#==============================================================================
#PlanningAgent implementations for the three experimental conditions.
#==============================================================================
#
#  LlamaBareAgent     — Method 1: vanilla Llama 3.1 8B Instruct.
#                        Input: procedure + history.       (NO candidate list)
#                        Output: free-form action name.
#                        "Does Llama plan procedures unaided?"
#
#  LlamaActionsAgent  — Method 2: vanilla Llama 3.1 8B Instruct.
#                        Input: procedure + history + candidate list.
#                        Output: one entry from the list (fuzzy-matched).
#                        "Does giving the model a closed action set help?"
#
#  PRMAgent           — Method 3: Llama 3.1 8B + PRM LoRA adapter.
#                        Input: procedure + history + candidate list.
#                        Output: candidate with the highest P(Yes) from PRM logits.
#                        "Does the fine-tuned PRM beat Llama as the agent?"
#
#All Llama-based agents default to Llama 3.1 8B Instruct in 4-bit (bitsandbytes
#nf4), matching the PRM training setup and fitting in ~6 GB VRAM.
#
#Each pick() returns (picked_action_name, info_dict). info_dict carries
#diagnostics that the runner merges into the saved trace step:
#  - Llama agents -> {"raw_response": <full model output before cleanup>}
#  - PRM agent    -> {"scores": {candidate: P(Yes), ...}}
#
#Heavy deps (torch / transformers / peft) are imported INSIDE each class on the
#first .pick() call so importing this module is cheap.
#==============================================================================

import math
import re
from difflib import SequenceMatcher
from pathlib import Path

from runner import PlanningAgent


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PRM_ADAPTER = Path(__file__).parent.parent / "PRM" / "trained_model"

#Same wording the PRM was trained on — see PRM/prepare_prm_data.py
PRM_SYSTEM_PROMPT = (
    "You are a process reward model for procedural workflows. "
    "Given a procedure description, the full list of available actions, "
    "and the steps completed so far, decide whether the proposed next action "
    "is correct at this point in the procedure. "
    "Answer only \"Yes\" or \"No\"."
)


# ---------------------------------------------------------------------------
# Map free-form model output to the closest candidate.
# ---------------------------------------------------------------------------
# Why we need this:
#   The model is told to reply with the action name "exactly as written", but in
#   practice it sometimes adds words ("Action: Submit Form"), changes capitals,
#   or drops words ("Submit"). We map back to the closest candidate so the saved
#   `picked` field is always one of the entries in `candidate_names`.
#
# Example:
#   response   = "Action: review the form"
#   candidates = ["Submit Form", "Review Form", "Approve Form"]
#   step 1 (substring): "review form" ⊂ "action: review the form"? No, the
#                       order differs. "review form" not found verbatim.
#   step 2 (ratio):     SequenceMatcher ratios are
#                         "submit form" → 0.42
#                         "review form" → 0.78  ← winner
#                         "approve form"→ 0.55
#   returns "Review Form"
def _best_candidate_match(response: str, candidates: list[str]) -> str:
    rl = response.lower().strip()
    # 1) case-insensitive substring match — fast path when model echoed the candidate
    for c in candidates:
        cl = c.lower()
        if cl in rl or rl.startswith(cl):
            return c
    # 2) fall back to character-level similarity ratio
    best, best_score = candidates[0], -1.0
    for c in candidates:
        score = SequenceMatcher(None, rl, c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best


# ---------------------------------------------------------------------------
# Shared base class for Methods 1 and 2 — loads Llama 3.1 8B Instruct in 4-bit.
# The model is loaded LAZILY on the first .pick() call so importing this module
# does not pin GPU memory.
# ---------------------------------------------------------------------------
class _LlamaInstructBase(PlanningAgent):
    def __init__(self, model_name: str = DEFAULT_BASE_MODEL, max_new_tokens: int = 64):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        # Idempotent — calling it twice is a no-op
        if self._model is not None:
            return
        # Defer the heavy imports until we actually need a GPU
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # 4-bit nf4 with bf16 compute — same recipe used in train_prm.py:109-114.
        # ~6 GB VRAM for Llama 3.1 8B; matches what the PRM was fine-tuned on.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading {self.model_name} (4-bit) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb,
            device_map="auto",  # let HF place layers across the available GPU(s)
        )
        self._model.eval()  # disable dropout / training-only behaviour
        print("  Model loaded.")

    def _generate(self, messages: list[dict]) -> str:
        # Greedy decode (do_sample=False, no temperature) so runs are reproducible.
        # max_new_tokens=64 is plenty for a single action name.
        #
        # Walk-through:
        #   messages = [{"role":"system","content":"You are a planner..."},
        #               {"role":"user","content":"Procedure: ...\n..."}]
        #
        #   apply_chat_template builds the Llama-3 prompt with proper headers
        #   and (importantly) appends the assistant turn header at the end via
        #   add_generation_prompt=True so the model continues with the answer.
        #
        #   We then slice off the input length to get only the new tokens.
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# METHOD 1 — vanilla Llama, no candidate list, free-form output.
# The hardest condition to score: the model can output anything.
# ---------------------------------------------------------------------------
# Example prompt the model receives:
#   SYSTEM: You are a procedural workflow planner.
#   USER:   Procedure: To start, reach out to the 1st Level Support...
#
#           Steps completed so far: Reach out to the 1st Level Support
#
#           What action should be done next? Reply with just the action name
#           (a short phrase, no full sentences).
#
# Example raw response: "Provide feedback to the account manager."
# After cleanup       : "Provide feedback to the account manager"
class LlamaBareAgent(_LlamaInstructBase):
    def pick(self, procedure_text, completed_names, candidate_names):
        self._load()
        steps_str = " → ".join(completed_names) if completed_names else "(none)"
        user = (
            f"Procedure: {procedure_text}\n\n"
            f"Steps completed so far: {steps_str}\n\n"
            "What action should be done next? Reply with just the action name "
            "(a short phrase, no full sentences)."
        )
        raw = self._generate([
            {"role": "system", "content": "You are a procedural workflow planner."},
            {"role": "user", "content": user},
        ])
        # Models sometimes prefix with "Next action: " or wrap in quotes.
        # Strip those decorations and keep only the first line.
        cleaned = re.sub(r"^[Nn]ext action:\s*", "", raw).strip(" \"'.")
        cleaned = cleaned.split("\n")[0].strip()
        return cleaned, {"raw_response": raw}


# ---------------------------------------------------------------------------
# METHOD 2 — vanilla Llama, given the candidate list, picks one entry.
# ---------------------------------------------------------------------------
# Example prompt the model receives:
#   SYSTEM: You are a procedural workflow planner. Pick exactly one action
#           from the available list as the next step.
#   USER:   Procedure: To start, reach out to the 1st Level Support...
#
#           Available actions:
#           - Reach out to the 1st Level Support
#           - Provide feedback to the account manager
#           - Ask a developer for assistance
#           - Provide feedback to the 1st Level Support
#
#           Steps completed so far: Reach out to the 1st Level Support
#
#           Which action should be done next? Reply with just the action name
#           exactly as written in the list above.
#
# raw     = "Ask a developer for assistance"
# matched = "Ask a developer for assistance"   (substring path of _best_candidate_match)
class LlamaActionsAgent(_LlamaInstructBase):
    def pick(self, procedure_text, completed_names, candidate_names):
        if not candidate_names:
            raise ValueError("LlamaActionsAgent requires a candidate list.")
        self._load()
        steps_str = " → ".join(completed_names) if completed_names else "(none)"
        actions_block = "\n".join(f"- {c}" for c in candidate_names)
        user = (
            f"Procedure: {procedure_text}\n\n"
            f"Available actions:\n{actions_block}\n\n"
            f"Steps completed so far: {steps_str}\n\n"
            "Which action should be done next? Reply with just the action name "
            "exactly as written in the list above."
        )
        raw = self._generate([
            {"role": "system", "content":
                "You are a procedural workflow planner. Pick exactly one action "
                "from the available list as the next step."},
            {"role": "user", "content": user},
        ])
        # Map paraphrases back to a canonical candidate
        matched = _best_candidate_match(raw, candidate_names)
        return matched, {"raw_response": raw}


# ---------------------------------------------------------------------------
# METHOD 3 — Llama 3.1 8B base + PRM LoRA adapter.
# ---------------------------------------------------------------------------
# How it works (per step):
#   for each candidate c in candidate_names:
#       prompt = build the same training-time PRM prompt (procedure +
#                available_actions + steps_so_far + "Proposed next action: c")
#       run a SINGLE forward pass; read logits[-1] for the next token
#       extract logits at "Yes" and "No" token IDs, softmax → P(Yes)
#   pick the candidate with the highest P(Yes).
#
# Why use logits and not generate("Yes")/("No"):
#   logits give a continuous score we can sort by. Generation only tells us
#   which is more likely (Yes-vs-No) — same answer for every candidate above
#   the 50% line, useless for ranking.
class PRMAgent(PlanningAgent):
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL,
                 adapter_path: Path = DEFAULT_PRM_ADAPTER,
                 system_prompt: str = PRM_SYSTEM_PROMPT):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.system_prompt = system_prompt
        self._model = None
        self._tokenizer = None
        # The token IDs for the strings "Yes" and "No" — set at load time.
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Same 4-bit recipe as training. Loading with different quantization here
        # would change the numerics of the LoRA forward pass.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading PRM: base={self.base_model}, adapter={self.adapter_path}")
        # The tokenizer was saved alongside the LoRA adapter at training time —
        # we use it (not the base model's tokenizer) so any chat-template tweaks
        # made during training are preserved.
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # 1) load the quantized base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb,
            device_map="auto",
        )
        # 2) attach the PRM LoRA on top
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
        # The chat template ends with the assistant turn header followed by "\n\n",
        # so the very next token starts the answer with no leading space.
        # We encode "Yes" and "No" without special tokens to get the token IDs.
        self._yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"  PRM loaded. yes_id={self._yes_id}, no_id={self._no_id}")

    def _score(self, procedure: str, available: list[str],
               completed: list[str], candidate: str) -> float:
        # Build the SAME prompt format the PRM was trained on
        # (see PRM/prepare_prm_data.py:79-85). Any deviation here will hurt
        # accuracy because the model has memorised this exact layout.
        #
        # Example user_content for one candidate:
        #   Procedure: To start, reach out to the 1st Level Support...
        #
        #   Available actions: Reach out to the 1st Level Support | Provide feedback ... | Ask a developer ...
        #
        #   Steps completed so far: Reach out to the 1st Level Support
        #
        #   Proposed next action: Ask a developer for assistance
        #
        #   Is this the correct next step?
        import torch
        steps_str = " → ".join(completed) if completed else "(none)"
        actions_str = " | ".join(available)
        user_content = (
            f"Procedure: {procedure}\n\n"
            f"Available actions: {actions_str}\n\n"
            f"Steps completed so far: {steps_str}\n\n"
            f"Proposed next action: {candidate}\n\n"
            f"Is this the correct next step?"
        )
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # Forward pass; logits[0, -1] is the next-token distribution
        # at the position just before the model would emit "Yes" or "No".
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1]
        yes = logits[self._yes_id].item()
        no = logits[self._no_id].item()

        # Two-way softmax over Yes/No only — equivalent to P(Yes) / (P(Yes) + P(No)).
        # Subtracting max(yes, no) keeps the exponentials numerically stable.
        # Example:  yes_logit = 12.4, no_logit = 8.2
        #           m = 12.4
        #           ey = exp(0)    = 1.0
        #           en = exp(-4.2) ≈ 0.015
        #           score = 1.0 / 1.015 = 0.985    → very confident "yes"
        m = max(yes, no)
        ey, en = math.exp(yes - m), math.exp(no - m)
        return ey / (ey + en)

    def pick(self, procedure_text, completed_names, candidate_names):
        # One forward pass per candidate. For a procedure with N predicted
        # actions and L steps, that's N×L forward passes total. With Flash-
        # Attention 2 + 4-bit it runs at ~3-5 candidates per second.
        if not candidate_names:
            raise ValueError("PRMAgent requires a candidate list.")
        self._load()
        scores: dict[str, float] = {}
        best_c, best_s = candidate_names[0], -1.0
        for c in candidate_names:
            s = self._score(procedure_text, candidate_names, completed_names, c)
            scores[c] = s
            if s > best_s:
                best_c, best_s = c, s
        # The full scores dict is returned so the saved trace shows ALL candidate
        # confidences at every step — invaluable when manually validating.
        return best_c, {"scores": scores}
