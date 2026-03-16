from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_length: int = 6144
    load_in_4bit: bool = True
    dtype: str = None


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: bool = True
