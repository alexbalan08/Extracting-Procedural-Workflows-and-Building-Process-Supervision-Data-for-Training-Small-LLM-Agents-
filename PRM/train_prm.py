
#Fine-tunes Llama 3.1 8B as a Process Reward Model (PRM) using SFT on step-level
#Yes/No labels. The model learns to score whether a proposed next action is correct
#given the procedure, available actions, and steps completed so far.
#
#At inference, run the model on each candidate action and extract
#P("Yes") / (P("Yes") + P("No")) from the logits to get a continuous score.
#Rank all candidate actions by this score and pick the highest — this is the re-ranker.
#
#Training setup mirrors the extractor SFT:
#  - Llama 3.1 8B, 4-bit quantization (bitsandbytes), LoRA r=16
#  - batch_size=1, gradient_accumulation=4 (effective batch=4) for 16GB VRAM
#  - max_seq_length=5120

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

_here = Path(__file__).parent


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_length: int = 5120
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")


def load_prm_dataset(sft_path: Path, tokenizer) -> Dataset:
    records = []
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    #apply the chat template so each record becomes a single formatted string
    #the model only trains on the assistant turn (Yes/No) via DataCollatorForCompletionOnlyLM
    texts = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


def train(sft_data_path=None, output_dir=None, model_cfg=None, lora_cfg=None,
          num_epochs=3, lr=2e-4, max_seq_length=None):

    model_cfg = model_cfg or ModelConfig()
    lora_cfg  = lora_cfg  or LoRAConfig()
    if max_seq_length is None:
        max_seq_length = model_cfg.max_seq_length

    sft_data_path = sft_data_path or (_here / "prm_sft_train.jsonl")
    output_dir    = output_dir    or (_here / "prm_model_output")

    if UNSLOTH_AVAILABLE:
        #unsloth gives ~2x speedup and uses Flash Attention 2 automatically
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=list(lora_cfg.target_modules),
            bias="none",
        )
    else:
        #fallback: standard transformers + bitsandbytes + peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_cfg.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=list(lora_cfg.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_prm_dataset(sft_data_path, tokenizer)

    #only compute loss on the assistant turn (the Yes/No token)
    #everything before that is the prompt and we dont want to train on it
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        #batch_size=1 with gradient_accumulation=4 gives effective batch of 4
        #this avoids OOM on 16GB VRAM with 5120-token sequences
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"PRM model saved to {output_dir}")


if __name__ == "__main__":
    train()
