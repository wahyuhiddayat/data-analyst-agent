import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Attention/MLP projections for Qwen-style models.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_hf_conversation(example: dict) -> tuple[list[dict], list[dict]]:
    """
    Convert a stored trajectory into the message shape the chat template expects.

    Tool-call arguments are stored as JSON strings (OpenAI format); the template
    serializes them itself, so parse them back into objects here.
    """
    messages = []
    for message in example["messages"]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            tool_calls = []
            for call in message["tool_calls"]:
                arguments = call["function"]["arguments"]
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    pass
                tool_calls.append({
                    "type": "function",
                    "function": {"name": call["function"]["name"], "arguments": arguments},
                })
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })
        else:
            messages.append(message)
    return messages, example["tools"]


def build_dataset(data_path: str, tokenizer, max_length: int):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize(example):
        messages, tools = to_hf_conversation(example)
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
        return tokenizer(text, truncation=True, max_length=max_length)

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def load_model(model_id: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto"
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune a student on teacher trajectories.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model id.")
    parser.add_argument("--data", required=True, help="Formatted training JSONL.")
    parser.add_argument("--out", required=True, help="Directory to save the LoRA adapter.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    ))
    model.print_trainable_parameters()

    dataset = build_dataset(args.data, tokenizer, args.max_length)
    print(f"Training examples: {len(dataset)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
