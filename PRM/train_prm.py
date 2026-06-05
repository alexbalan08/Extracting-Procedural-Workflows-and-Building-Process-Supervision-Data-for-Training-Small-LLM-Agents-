
#Llama 3.1 8B as PRM using our sft created data
#Yes/No labels so the model learns to score whether a proposed next action is correct
#given the procedure, available actions, and steps completed so far.

#at inference time 
#P("Yes") / (P("Yes") + P("No")) from the logits to get a continuous score.
#Rank all candidate actions by this score and pick the highest or see what we do later with this score 
#that we actually combine it with llama frozen next token proabbailtiies


#!!this training script is entirelly generated with claude!! giving the desired parameters

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

_here = Path(__file__).parent

MAX_SEQ_LENGTH = 5120


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")


def load_and_tokenize(sft_path: Path, tokenizer) -> Dataset:
    records = []
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    #the PRM is read from a single answer token ("Yes"/"No"), so train ONLY on the assistant answer:
    #tokenize the full conversation, then mask every prompt token (system+user+assistant header) to -100
    #so the loss ignores it. without this the loss is dominated by hundreds of prompt tokens per answer.
    examples = []
    for rec in records:
        messages = rec["messages"]
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
        #same conversation minus the answer, ending at the assistant header — this is the prompt prefix
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
        )

        #defensively take only the genuinely shared prefix (guards against a tokenizer merging the
        #header/answer boundary), so we never mask into the answer or leave prompt tokens unmasked
        n_prompt = len(prompt_ids)
        while n_prompt > 0 and full_ids[:n_prompt] != prompt_ids[:n_prompt]:
            n_prompt -= 1

        full_ids = full_ids[:MAX_SEQ_LENGTH]
        n_prompt = min(n_prompt, len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]

        examples.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })
    return Dataset.from_list(examples)


def train(sft_data_path=None, output_dir=None, model_cfg=None, lora_cfg=None,
          num_epochs=3, lr=2e-4):

    model_cfg = model_cfg or ModelConfig()
    lora_cfg  = lora_cfg  or LoRAConfig()

    sft_data_path = sft_data_path or (_here / "prm_sft_train.jsonl")
    output_dir    = output_dir    or str(_here / "prm_model_output")

    if UNSLOTH_AVAILABLE:
        #unsloth gives ~2x speedup and uses Flash Attention 2 automatically
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.model_name,
            max_seq_length=MAX_SEQ_LENGTH,
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

    dataset = load_and_tokenize(sft_data_path, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        #pads input_ids/attention_mask and pads labels with -100 (preserving our prompt masking)
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    #resume only if there's actually a checkpoint subdir — Path.exists() is True for an empty
    #pre-created directory (common on Colab when mounting Drive first) and would crash with
    #"Can't find a valid checkpoint" otherwise
    has_ckpt = Path(output_dir).exists() and any(Path(output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=has_ckpt)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"PRM model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PRM via SFT on step-level Yes/No labels")
    parser.add_argument("--sft_data", type=str, default=None, help="Path to prm_sft_train.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Path to save model (use gdrive path on Colab)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train(
        sft_data_path=Path(args.sft_data) if args.sft_data else None,
        output_dir=args.output,
        num_epochs=args.epochs,
        lr=args.lr,
    )
