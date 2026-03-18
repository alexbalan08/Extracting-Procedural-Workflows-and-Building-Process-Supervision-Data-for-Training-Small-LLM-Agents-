# SFT training script for fine-tuning LLM on procedure extraction task with LoRA

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from training_config import ModelConfig, LoRAConfig

DEFAULT_SFT_DATA = Path(__file__).parent.parent / "data_prep" / "sft_train.jsonl"


def load_model(model_cfg: ModelConfig, lora_cfg: LoRAConfig):
    # load base model with 4-bit quantization
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
    # prepare for training on quantized model
    model = prepare_model_for_kbit_training(model)
    # apply LoRA adapters
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg.r, lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout, bias=lora_cfg.bias,
        target_modules=lora_cfg.target_modules, task_type="CAUSAL_LM",
    ))
    return model, tokenizer


def train(sft_data_path, output_dir, model_cfg=None, lora_cfg=None,
          num_epochs=3, lr=2e-4, max_seq_length=None):
    # load config with defaults
    model_cfg = model_cfg or ModelConfig()
    lora_cfg = lora_cfg or LoRAConfig()
    if max_seq_length is None:
        max_seq_length = model_cfg.max_seq_length

    print(f"Loading model: {model_cfg.model_name}")
    model, tokenizer = load_model(model_cfg, lora_cfg)

    # load and format dataset with chat templates
    print(f"Loading dataset from {sft_data_path}")
    dataset = Dataset.from_json(str(sft_data_path))
    dataset = dataset.map(
        lambda ex: {"text": [
            tokenizer.apply_chat_template(m, tokenize=False) for m in ex["messages"]
        ]},
        batched=True, num_proc=4, remove_columns=dataset.column_names,
    )

    # setup SFT trainer with config
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_length=max_seq_length,
            packing=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # batch size 2 for colab A40
            gradient_accumulation_steps=4,  # effective batch size = 8
            learning_rate=lr,
            warmup_steps=40,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            save_steps=100,
            logging_steps=10,
            seed=42,
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        ),
    )

    print("Starting training …")
    trainer.train()

    # save adapter weights + tokenizer
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", type=Path, default=DEFAULT_SFT_DATA)
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # override model name if provided
    model_cfg = ModelConfig()
    if args.model:
        model_cfg.model_name = args.model

    train(args.sft_data, args.output, model_cfg=model_cfg,
          num_epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()

