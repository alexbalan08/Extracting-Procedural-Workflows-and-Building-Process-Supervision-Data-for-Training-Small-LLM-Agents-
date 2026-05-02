#PlanningAgent implementations for the three experimental conditions.
#LlamaBareAgent is method 1, vanilla llama with no candidate list, free-form output.
#LlamaActionsAgent is method 2, vanilla llama given the candidate list, picks one.
#EnsemblePlannerAgent is method 3, base llama plus the prm lora blended at inference.
#all llama-based agents default to llama 3.1 8b instruct in 4-bit (bitsandbytes nf4) so
#everything fits in ~6 GB VRAM and matches the prm training recipe.
#each pick() returns (picked_action_name, info_dict). info_dict carries diagnostics that
#the runner merges into the saved trace step.
#heavy deps (torch / transformers / peft) are imported inside each class on the first
#pick() call so importing this module is cheap.

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


#maps a free-form model output back to the closest candidate string
#the model sometimes adds words or changes capitalisation so we substring-match first
#then fall back to character similarity. picked is always one of the candidate_names
def _best_candidate_match(response: str, candidates: list[str]) -> str:
    rl = response.lower().strip()
    #substring match first — fast path when the model echoed the candidate
    for c in candidates:
        cl = c.lower()
        if cl in rl or rl.startswith(cl):
            return c
    #fall back to character-level similarity ratio
    best, best_score = candidates[0], -1.0
    for c in candidates:
        score = SequenceMatcher(None, rl, c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best


#shared base for methods 1 and 2 that loads llama 3.1 8b instruct in 4-bit
#the model is loaded lazily on the first pick() call so importing this module is cheap
class _LlamaInstructBase(PlanningAgent):
    def __init__(self, model_name: str = DEFAULT_BASE_MODEL, max_new_tokens: int = 64):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        #same 4-bit nf4 recipe as the prm so loading numerics match
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
            device_map="auto",
        )
        self._model.eval()
        print("  Model loaded.")

    def _generate(self, messages: list[dict]) -> str:
        #greedy decode so runs are reproducible. 64 new tokens is enough for one action name
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


#method 1 vanilla llama with no candidate list, free-form output
#hardest condition since the model can say anything
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
        #strip "Next action:" prefixes and quote/period decorations, keep only the first line
        cleaned = re.sub(r"^[Nn]ext action:\s*", "", raw).strip(" \"'.")
        cleaned = cleaned.split("\n")[0].strip()
        return cleaned, {"raw_response": raw}


#method 2 vanilla llama given the candidate list, picks one entry
#raw output is mapped back to a canonical candidate via _best_candidate_match
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
        #map paraphrases back to a canonical candidate
        matched = _best_candidate_match(raw, candidate_names)
        return matched, {"raw_response": raw}


#method 3 base llama 3.1 8b plus the prm lora ensembled for next-action picking
#we load the base model once and attach the prm lora on top
#for each candidate we score it twice on the same model
#prm score with the lora ON gives yes/no logits softmaxed to P(yes)
#llm score with the lora OFF (via peft disable_adapter) gives the mean log-prob of generating the candidate
#we softmax-normalise each scorer over the candidate set then blend with weight alpha and pick argmax
#alpha=1.0 is pure prm, alpha=0.0 is pure base llama, default 0.5 is equal blend
#llm_temp flattens the base llama distribution if it gets too peaky on a single option
class EnsemblePlannerAgent(PlanningAgent):
    LLM_SYSTEM_PROMPT = (
        "You are a procedural workflow planner. "
        "Given the procedure description and the steps completed so far, "
        "predict the next action."
    )

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL,
                 adapter_path: Path = DEFAULT_PRM_ADAPTER,
                 system_prompt: str = PRM_SYSTEM_PROMPT,
                 alpha: float = 0.5,
                 llm_temp: float = 1.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.system_prompt = system_prompt
        self.alpha = alpha
        self.llm_temp = llm_temp
        self._model = None
        self._tokenizer = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        #same 4-bit recipe as training, otherwise the lora forward numerics drift
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading: base={self.base_model}, adapter={self.adapter_path}")
        #tokenizer comes from the adapter dir to keep any chat-template tweaks made at training
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, quantization_config=bnb, device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
        #Yes / No token IDs for the prm logit read
        self._yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"  loaded. yes_id={self._yes_id}, no_id={self._no_id}")

    def _score_prm(self, procedure: str, available: list[str],
                   completed: list[str], candidate: str) -> float:
        #same prompt format as PRM training (see PRM/prepare_prm_data.py)
        #lora must be ON for this call
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
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1]
        yes = logits[self._yes_id].item()
        no = logits[self._no_id].item()
        #two-way softmax over Yes/No → P(Yes), numerically stable
        m = max(yes, no)
        ey, en = math.exp(yes - m), math.exp(no - m)
        return ey / (ey + en)

    def _score_llm(self, procedure: str, completed: list[str],
                   candidate: str) -> float:
        #length-normalised mean log-prob of the candidate string under the base llama
        #lora is toggled OFF only for this forward pass via peft disable_adapter context
        import torch
        import torch.nn.functional as F

        steps_str = " → ".join(completed) if completed else "(none)"
        user_content = (
            f"Procedure: {procedure}\n\n"
            f"Steps completed so far: {steps_str}\n\n"
            f"What is the next action?"
        )
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": self.LLM_SYSTEM_PROMPT},
             {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        full = prompt + candidate

        #tokenise both so we know exactly where the candidate tokens start
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        full_ids   = self._tokenizer(full,   return_tensors="pt").input_ids.to(self._model.device)

        with torch.no_grad(), self._model.disable_adapter():
            logits = self._model(full_ids).logits[0]

        prompt_len = prompt_ids.shape[1]
        cand_ids = full_ids[0, prompt_len:]
        if len(cand_ids) == 0:
            return 0.0
        #logits[i] predicts token i+1, so to score cand_ids[j] read logits at prompt_len+j-1
        cand_logits = logits[prompt_len - 1 : -1]
        log_probs = F.log_softmax(cand_logits, dim=-1)
        chosen = log_probs[range(len(cand_ids)), cand_ids]
        return (chosen.sum() / len(cand_ids)).item()

    @staticmethod
    def _softmax_dist(xs: list[float], temp: float = 1.0):
        #numerically stable softmax over a small list of scalars, returns numpy array
        import numpy as np
        a = np.array(xs, dtype=np.float64) / max(temp, 1e-6)
        a = a - a.max()
        e = np.exp(a)
        return e / e.sum()

    def pick(self, procedure_text, completed_names, candidate_names):
        if not candidate_names:
            raise ValueError("EnsemblePlannerAgent requires a candidate list.")
        self._load()

        prm_raw: list[float] = []
        llm_raw: list[float] = []
        for c in candidate_names:
            prm_raw.append(self._score_prm(procedure_text, candidate_names, completed_names, c))
            llm_raw.append(self._score_llm(procedure_text, completed_names, c))

        #softmax each scorer over the candidate set so both live on the same scale
        prm_dist = self._softmax_dist(prm_raw, temp=1.0)
        llm_dist = self._softmax_dist(llm_raw, temp=self.llm_temp)
        final    = self.alpha * prm_dist + (1.0 - self.alpha) * llm_dist

        import numpy as np
        best_idx = int(np.argmax(final))
        #keep raw and normalised scores per candidate for post-hoc analysis
        return candidate_names[best_idx], {
            "alpha":      self.alpha,
            "llm_temp":   self.llm_temp,
            "prm_scores": dict(zip(candidate_names, prm_raw)),
            "llm_scores": dict(zip(candidate_names, llm_raw)),
            "prm_dist":   dict(zip(candidate_names, prm_dist.tolist())),
            "llm_dist":   dict(zip(candidate_names, llm_dist.tolist())),
            "final":      dict(zip(candidate_names, final.tolist())),
        }
