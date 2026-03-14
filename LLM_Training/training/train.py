

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from training_config import ModelConfig, LoRAConfig

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

DEFAULT_SFT_DATA = Path(__file__).parent.parent / "data_prep" / "sft_train.jsonl"


def load_model(model_cfg: ModelConfig, lora_cfg: LoRAConfig):
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.model_name, max_seq_length=model_cfg.max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit, dtype=model_cfg.dtype,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=lora_cfg.r, lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout, bias=lora_cfg.bias,
            target_modules=lora_cfg.target_modules,
            use_gradient_checkpointing=lora_cfg.use_gradient_checkpointing,
            random_state=42,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name, quantization_config=bnb_config, device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg.r, lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout, bias=lora_cfg.bias,
            target_modules=lora_cfg.target_modules, task_type="CAUSAL_LM",
        ))
    return model, tokenizer


def train(sft_data_path, output_dir, model_cfg=None, lora_cfg=None,
          num_epochs=3, lr=2e-4, max_seq_length=None):
    model_cfg = model_cfg or ModelConfig()
    lora_cfg = lora_cfg or LoRAConfig()
    if max_seq_length is None:
        max_seq_length = model_cfg.max_seq_length

    print(f"Loading model: {model_cfg.model_name}")
    model, tokenizer = load_model(model_cfg, lora_cfg)

    print(f"Loading dataset from {sft_data_path}")
    dataset = Dataset.from_json(str(sft_data_path))
    dataset = dataset.map(
        lambda ex: {"text": [
            tokenizer.apply_chat_template(m, tokenize=False) for m in ex["messages"]
        ]},
        batched=True, num_proc=4, remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset, dataset_text_field="text",
        max_seq_length=max_seq_length, packing=False,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=True,
            max_grad_norm=0.3,
            save_steps=100,
            logging_steps=10,
            seed=42,
            report_to="none",
        ),
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", type=Path, default=DEFAULT_SFT_DATA)
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    if args.model:
        model_cfg.model_name = args.model

    train(args.sft_data, args.output, model_cfg=model_cfg,
          num_epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
