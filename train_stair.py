
"""
Finance STAIR training using Mistral 7B + QLoRA.

Input:
    data/stair_train.jsonl

Output:
    artifacts/stair-finance-lora

Expected record:
{
    "question": "...",
    "answer": "...",
    "gold_leaf": "...",
    "toc": ["Section 1", "Section 2"]
}
"""

import json
import argparse
import inspect

from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


# ---------------------------------------------------------
# 1. ToC utilities
# ---------------------------------------------------------

def flatten_toc(toc):

    leaves = []

    def visit(node):

        if isinstance(node, str):
            leaves.append(node.strip())
            return

        if not isinstance(node, dict):
            return

        title = str(node.get("title", "")).strip()

        children = (
            node.get("children")
            or node.get("subsections")
            or node.get("sections")
            or []
        )

        if children:
            for child in children:
                visit(child)
        elif title:
            leaves.append(title)

    for item in toc or []:
        visit(item)

    result = []
    seen = set()

    for title in leaves:

        key = title.lower()

        if title and key not in seen:
            seen.add(key)
            result.append(title)

    return result


# ---------------------------------------------------------
# 2. Exact training prompt
# ---------------------------------------------------------

def build_prompt(question, toc):

    titles = flatten_toc(toc)

    toc_string = "\n".join(
        f"- {title}"
        for title in titles
    )

    return (
        "You are a structure-aware financial document retriever.\n"
        "Given a financial question and the document table of contents, "
        "select the single most relevant leaf section.\n"
        "Output ONLY one exact section title from the supplied ToC.\n\n"
        f"TABLE OF CONTENTS:\n{toc_string}\n\n"
        f"QUESTION: {question}\n"
        "SECTION:"
    )


# ---------------------------------------------------------
# 3. Load and validate training records
# ---------------------------------------------------------

def load_records(train_file):

    rows = []

    with open(
        train_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, 1):

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid JSON at line {line_number}"
                )
                continue

            question = str(
                record.get("question", "")
            ).strip()

            gold_leaf = str(
                record.get("gold_leaf", "")
            ).strip()

            toc = record.get("toc", [])

            titles = flatten_toc(toc)

            if not question or not gold_leaf or not titles:
                continue

            # Ensure the target is a valid ToC leaf.
            matched = next(
                (
                    title for title in titles
                    if title.lower() == gold_leaf.lower()
                ),
                None
            )

            if matched is None:
                print(
                    f"Skipping example {line_number}: "
                    f"gold_leaf is not in the ToC: {gold_leaf}"
                )
                continue

            rows.append({
                "question": question,
                "gold_leaf": matched,
                "toc": titles
            })

    return rows


# ---------------------------------------------------------
# 4. Tokenized dataset with prompt masking
# ---------------------------------------------------------

class STAIRDataset(Dataset):

    def __init__(
        self,
        rows,
        tokenizer,
        max_length=2048
    ):

        self.items = []

        for row in rows:

            prompt = build_prompt(
                row["question"],
                row["toc"]
            )

            completion = " " + row["gold_leaf"]

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length - 80
            )["input_ids"]

            completion_ids = tokenizer(
                completion + tokenizer.eos_token,
                add_special_tokens=False
            )["input_ids"]

            # Ensure the completion is not cut off.
            if len(prompt_ids) + len(completion_ids) > max_length:

                available = max_length - len(completion_ids)

                if available <= 0:
                    continue

                prompt_ids = prompt_ids[:available]

            input_ids = prompt_ids + completion_ids

            # Ignore prompt tokens when calculating training loss.
            labels = (
                [-100] * len(prompt_ids)
                + completion_ids
            )

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# ---------------------------------------------------------
# 5. Split training and validation data
# ---------------------------------------------------------

def split_records(rows, validation_ratio=0.1):

    if len(rows) < 2:
        return rows, []

    # Fixed seed makes the split reproducible.
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(rows),
        generator=generator
    ).tolist()

    validation_size = max(
        1,
        int(len(rows) * validation_ratio)
    )

    validation_size = min(
        validation_size,
        len(rows) - 1
    )

    validation_indices = set(
        indices[:validation_size]
    )

    train_rows = [
        row for i, row in enumerate(rows)
        if i not in validation_indices
    ]

    validation_rows = [
        row for i, row in enumerate(rows)
        if i in validation_indices
    ]

    return train_rows, validation_rows


# ---------------------------------------------------------
# 6. Main training function
# ---------------------------------------------------------

def main(
    train_file="data/stair_train.jsonl",
    output="artifacts/stair-finance-lora",
    epochs=2,
    max_length=2048,
    batch_size=1,
    learning_rate=2e-4
):

    train_path = Path(train_file)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. Run this training script "
            "in Google Colab with a GPU runtime."
        )

    print("Loading training records...")

    rows = load_records(train_file)

    if len(rows) < 2:
        raise RuntimeError(
            "At least two valid training examples are required. "
            "Check data/stair_train.jsonl and the ToC labels."
        )

    train_rows, validation_rows = split_records(rows)

    print(f"Total examples: {len(rows)}")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # QLoRA quantization
    # -----------------------------------------------------

    print("Preparing 4-bit quantization...")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    # -----------------------------------------------------
    # Load base model
    # -----------------------------------------------------

    print("Loading Mistral 7B...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    # -----------------------------------------------------
    # LoRA configuration
    # -----------------------------------------------------

    print("Applying LoRA...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    # -----------------------------------------------------
    # Prepare datasets
    # -----------------------------------------------------

    print("Tokenizing datasets...")

    train_dataset = STAIRDataset(
        train_rows,
        tokenizer,
        max_length=max_length
    )

    validation_dataset = STAIRDataset(
        validation_rows,
        tokenizer,
        max_length=max_length
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No valid tokenized training examples were created."
        )

    if len(validation_dataset) == 0:
        validation_dataset = None

    print(f"Tokenized training examples: {len(train_dataset)}")

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    output_path = Path(output)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    training_parameters = {
        "output_dir": str(output_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "remove_unused_columns": False
    }

    # Support both newer and older Transformers versions.
    training_signature = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if validation_dataset is not None:

        if "eval_strategy" in training_signature:
            training_parameters["eval_strategy"] = "epoch"

        elif "evaluation_strategy" in training_signature:
            training_parameters["evaluation_strategy"] = "epoch"

        training_parameters["load_best_model_at_end"] = True
        training_parameters["metric_for_best_model"] = "eval_loss"
        training_parameters["greater_is_better"] = False

    training_args = TrainingArguments(
        **training_parameters
    )

    # -----------------------------------------------------
    # Data collator
    # -----------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator
    )

    # -----------------------------------------------------
    # Start training
    # -----------------------------------------------------

    print("\nStarting Finance STAIR training...\n")

    trainer.train()

    # -----------------------------------------------------
    # Save adapter and tokenizer
    # -----------------------------------------------------

    print("Saving trained LoRA adapter...")

    trainer.save_model(str(output_path))

    tokenizer.save_pretrained(str(output_path))

    print("\nTraining completed successfully.")
    print(f"LoRA adapter saved to: {output_path}")

    if validation_dataset is not None:

        print("\nValidation results:")

        metrics = trainer.evaluate()

        for key, value in metrics.items():
            print(f"{key}: {value}")


# ---------------------------------------------------------
# 7. Command line arguments
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file",
        default="data/stair_train.jsonl"
    )

    parser.add_argument(
        "--output",
        default="artifacts/stair-finance-lora"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4
    )

    args = parser.parse_args()

    main(
        train_file=args.train_file,
        output=args.output,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
