#Llama base is method 1, frozen llama with no candidate list, free form output.
#Llama with access to extracted actions is method 2 base llama given the candidate list, picks one
#a combined method between planner and llma agent is method 3, base llama plus the prm lora blended at inference.
#method 4 is method 3 plus an agentic tool call when the blended score is uncertain the agent looks
#up the extracted graph (execution_states from held_out.json made implicitly from the graphs) to narrow candidates to the valid-next set
#and re-picks. compare method 4 vs method 3 to measure what the graph tool adds and it should add a lot



import math
import re
from difflib import SequenceMatcher
from pathlib import Path

from runner import PlanningAgent


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PRM_ADAPTER = Path(__file__).parent.parent / "PRM" / "trained_model"

#Same wording the PRM was trained on
PRM_SYSTEM_PROMPT = (
    "You are a process reward model for procedural workflows. "
    "Given a procedure description, the full list of available actions, "
    "and the steps completed so far, decide whether the proposed next action "
    "is correct at this point in the procedure. "
    "Answer only \"Yes\" or \"No\"."
)


#maps a free-form model output back to the closest candidate string
#then fall back to character similarity. picked is always one of the candidate_names
def _best_candidate_match(response: str, candidates: list[str]) -> str:
    rl = response.lower().strip()
   
    for c in candidates:
        cl = c.lower()
        if cl in rl or rl.startswith(cl):
            return c
  
    best, best_score = candidates[0], -1.0
    for c in candidates:
        score = SequenceMatcher(None, rl, c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best


#shared base for methods 1 and 2 that loads llama 3.1 8b instruct in 4-bit
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


#method 1 
class LlamaBareAgent(_LlamaInstructBase):
    
    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
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
        
        cleaned = re.sub(r"^[Nn]ext action:\s*", "", raw).strip(" \"'.")
        cleaned = cleaned.split("\n")[0].strip()
        return cleaned, {"raw_response": raw}


#OpenAI baseline same prompt and behaviour as LlamaBareAgent but routes the call
#to the OpenAI API instead of the local llama
#requires OPENAI_API_KEY in the environment but we have it already from extractor

class OpenAIBareAgent(PlanningAgent):
    

    #we need to increase max tokens by a lot gor gpt 5.5 since it s a reasoning model which lawasys produceds cot in the output before
    def __init__(self, model: str = "gpt-5.5 ", max_completion_tokens: int = 4000):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Export it before running.")
        self._client = OpenAI(api_key=api_key)
        print(f"OpenAI client ready (model={self.model}).")

    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
        self._load()
        steps_str = " → ".join(completed_names) if completed_names else "(none)"
        user = (
            f"Procedure: {procedure_text}\n\n"
            f"Steps completed so far: {steps_str}\n\n"
            "What action should be done next? Reply with just the action name "
            "(a short phrase, no full sentences)."
        )
        #temperature is intentionally not set bcs newer reasoning models (gpt-5.5)
        #reject any value other than the default 1. seed=42 still
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a procedural workflow planner."},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=self.max_completion_tokens,
            seed=42,
        )
        raw = (response.choices[0].message.content or "").strip()
        cleaned = re.sub(r"^[Nn]ext action:\s*", "", raw).strip(" \"'.")
        cleaned = cleaned.split("\n")[0].strip()
        return cleaned, {"raw_response": raw}


#method 2 
class LlamaActionsAgent(_LlamaInstructBase):
    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
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


#OpenAI baseline given the candidate list same as method 2 but with openai 
class OpenAIActionsAgent(PlanningAgent):
    def __init__(self, model: str = "gpt-5.4-mini", max_completion_tokens: int = 4000):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Export it before running.")
        self._client = OpenAI(api_key=api_key)
        print(f"OpenAI client ready (model={self.model}).")

    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
        if not candidate_names:
            raise ValueError("OpenAIActionsAgent requires a candidate list.")
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
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content":
                    "You are a procedural workflow planner. Pick exactly one action "
                    "from the available list as the next step."},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=self.max_completion_tokens,
            seed=42,
        )
        raw = (response.choices[0].message.content or "").strip()
        matched = _best_candidate_match(raw, candidate_names)
        return matched, {"raw_response": raw}


#method 3 base llama 3.1 8b plus the prm lora ensembled for next-action picking
#we load the base model once and attach the prm lora on top
#for each candidate we score it twice on the same model
#we softmax-normalise each scorer over the candidate set then blend with weight alpha and pick argmax
#alpha=1.0 is pure prm, alpha=0.0 
class EnsemblePlannerAgent(PlanningAgent):
    LLM_SYSTEM_PROMPT = (
        "You are a procedural workflow planner. "
        "Given the procedure description and the steps completed so far, "
        "predict the next action."
    )

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL,
                 adapter_path: Path = DEFAULT_PRM_ADAPTER,
                 system_prompt: str = PRM_SYSTEM_PROMPT,
                 alpha: float = 0.90,
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

        
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading: base={self.base_model}, adapter={self.adapter_path}")
    
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, quantization_config=bnb, device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
      
        self._yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"  loaded. yes_id={self._yes_id}, no_id={self._no_id}")

    def _score_prm(self, procedure: str, available: list[str],
                   completed: list[str], candidate: str) -> float:

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
        #we need softmax over yes and no actions
        m = max(yes, no)
        ey, en = math.exp(yes - m), math.exp(no - m)
        return ey / (ey + en)

    def _score_llm(self, procedure: str, completed: list[str],
                   candidate: str) -> float:
        
        #length-normalised mean log-prob of the candidate string under the base llama
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

        
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        full_ids   = self._tokenizer(full,   return_tensors="pt").input_ids.to(self._model.device)

        with torch.no_grad(), self._model.disable_adapter():
            logits = self._model(full_ids).logits[0]

        prompt_len = prompt_ids.shape[1]
        cand_ids = full_ids[0, prompt_len:]
        if len(cand_ids) == 0:
            return 0.0
        
     
        cand_logits = logits[prompt_len - 1 : -1]
        log_probs = F.log_softmax(cand_logits, dim=-1)
        chosen = log_probs[range(len(cand_ids)), cand_ids]
        return (chosen.sum() / len(cand_ids)).item()

    @staticmethod
    def _softmax_dist(xs: list[float], temp: float = 1.0):
        
        import numpy as np
        a = np.array(xs, dtype=np.float64) / max(temp, 1e-6)
        a = a - a.max()
        e = np.exp(a)
        return e / e.sum()

    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
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
        #keep raw and normalised scores per candidate for manual inspection
        return candidate_names[best_idx], {
            "alpha":      self.alpha,
            "llm_temp":   self.llm_temp,
            "prm_scores": dict(zip(candidate_names, prm_raw)),
            "llm_scores": dict(zip(candidate_names, llm_raw)),
            "prm_dist":   dict(zip(candidate_names, prm_dist.tolist())),
            "llm_dist":   dict(zip(candidate_names, llm_dist.tolist())),
            "final":      dict(zip(candidate_names, final.tolist())),
        }


#method 4 method 3 plus an agentic graph-tool call when the agent is uncertain
#we run the same prm + llama blend first. if the top blended score is below tool_threshold AND
#the margin to runner-up is below tool_margin, we look up valid next actions in the predicted
#execution_states (the extracted graph) and re-pick from that narrower set

class AgenticEnsembleAgent(EnsemblePlannerAgent):
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL,
                 adapter_path: Path = DEFAULT_PRM_ADAPTER,
                 system_prompt: str = PRM_SYSTEM_PROMPT,
                 alpha: float = 0.5,
                 llm_temp: float = 1.0,
                 tool_threshold: float = 0.85,
                 tool_margin: float = 0.2):
        super().__init__(base_model, adapter_path, system_prompt, alpha, llm_temp)
        self.tool_threshold = tool_threshold
        self.tool_margin = tool_margin

    @staticmethod
    def _valid_next_from_graph(history_names, predicted_states, id_to_name):
     
        if not predicted_states or not id_to_name:
            return None
        name_to_id = {v: k for k, v in id_to_name.items()}
        history_ids = [name_to_id.get(n, n) for n in history_names]
        valid_ids = set()
        for s in predicted_states:
            if s.get("completed_actions") == history_ids:
                valid_ids.update(s.get("available_next", []))
        return [id_to_name[i] for i in valid_ids if i in id_to_name]

    def pick(self, procedure_text, completed_names, candidate_names,
             predicted_states=None, id_to_name=None):
        if not candidate_names:
            raise ValueError("AgenticEnsembleAgent requires a candidate list.")
        self._load()

        #initial pass score every candidate the same way method 3 does
        prm_raw: list[float] = []
        llm_raw: list[float] = []
        for c in candidate_names:
            prm_raw.append(self._score_prm(procedure_text, candidate_names, completed_names, c))
            llm_raw.append(self._score_llm(procedure_text, completed_names, c))

        prm_dist = self._softmax_dist(prm_raw, temp=1.0)
        llm_dist = self._softmax_dist(llm_raw, temp=self.llm_temp)
        final    = self.alpha * prm_dist + (1.0 - self.alpha) * llm_dist

        import numpy as np
        sorted_idx = list(np.argsort(-final))
        top_idx, second_idx = sorted_idx[0], (sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0])
        top_score = float(final[top_idx])
        margin = top_score - float(final[second_idx])

        info = {
            "alpha":          self.alpha,
            "llm_temp":       self.llm_temp,
            "tool_threshold": self.tool_threshold,
            "tool_margin":    self.tool_margin,
            "prm_scores":     dict(zip(candidate_names, prm_raw)),
            "llm_scores":     dict(zip(candidate_names, llm_raw)),
            "prm_dist":       dict(zip(candidate_names, prm_dist.tolist())),
            "llm_dist":       dict(zip(candidate_names, llm_dist.tolist())),
            "final":          dict(zip(candidate_names, final.tolist())),
            "top_score":      top_score,
            "margin":         margin,
        }

        #tool call either condition activates when the agent is unsure
        gate_fires = (top_score < self.tool_threshold) or (margin < self.tool_margin)
        if not gate_fires:
            info["tool_called"] = False
            return candidate_names[top_idx], info

        #consult the predicted graph for valid next actions
        valid_next = self._valid_next_from_graph(completed_names, predicted_states, id_to_name)
        narrowed = [c for c in candidate_names if valid_next and c in valid_next]

        info["tool_called"] = True
        info["valid_next"] = valid_next
        info["narrowed"] = narrowed

        if not narrowed or len(narrowed) == len(candidate_names):
            #tool returned nothing useful (no matching state, or full list) just fallback
            info["tool_useful"] = False
            return candidate_names[top_idx], info

        #re-rank within the narrowed set by reusing the existing  scores
        #no need to re score everything again since history andcompleted consitions are the same
        narrowed_indices = [i for i, c in enumerate(candidate_names) if c in narrowed]
        prm_n = [prm_raw[i] for i in narrowed_indices]
        llm_n = [llm_raw[i] for i in narrowed_indices]
        prm_dist_n = self._softmax_dist(prm_n, temp=1.0)
        llm_dist_n = self._softmax_dist(llm_n, temp=self.llm_temp)
        final_n = self.alpha * prm_dist_n + (1.0 - self.alpha) * llm_dist_n
        best_n = int(np.argmax(final_n))
        picked = candidate_names[narrowed_indices[best_n]]

        info["tool_useful"] = True
        info["narrowed_final"] = dict(zip(narrowed, final_n.tolist()))
        return picked, info
