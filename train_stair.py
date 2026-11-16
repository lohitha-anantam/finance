
"""
Finance STAIR - Mistral 7B QLoRA Training

Base model:
    mistralai/Mistral-7B-Instruct-v0.2

Training method:
    4-bit Quantization + LoRA

Input:
    data/stair_train.jsonl

Output:
    artifacts/stair-finance-lora
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


# --------------------------------------------------
# 1. Flatten Table of Contents
# --------------------------------------------------

def flatten_toc(toc):

    output = []

    def visit(node):

        if isinstance(node, str):

            if node.strip():
                output.append(node.strip())

            return

        if not isinstance(node, dict):
            return

        title = str(
            node.get("title", "")
        ).strip()

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

            output.append(title)

    for item in toc or []:
        visit(item)

    result = []
    seen = set()

    for title in output:

        key = title.lower()

        if key not in seen:

            seen.add(key)
            result.append(title)

    return result


# --------------------------------------------------
# 2. Build Training Prompt
# --------------------------------------------------

def build_prompt(question, toc):

    titles = flatten_toc(toc)

    formatted_toc = "\n".join(
        f"- {title}"
        for title in titles
    )

    return (
        "You are a structure-aware financial document retriever.\n"
        "Choose the single most relevant section title "
        "from the supplied list.\n"
        "Return exactly one title copied from the list. "
        "Do not return a sentence from the report.\n\n"
        f"TABLE OF CONTENTS:\n{formatted_toc}\n\n"
        f"QUESTION: {question}\n"
        "SECTION:"
    )


# --------------------------------------------------
# 3. Load and Validate Training Data
# --------------------------------------------------

def load_training_data(train_file):

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
                    f"Skipping invalid JSON line {line_number}"
                )

                continue

            question = str(
                record.get("question", "")
            ).strip()

            gold_leaf = str(
                record.get("gold_leaf", "")
            ).strip()

            toc = record.get(
                "toc",
                []
            )

            titles = flatten_toc(toc)

            if not question or not gold_leaf or not titles:
                continue

            # Ensure target exists in ToC.

            matched_title = next(

                (

                    title

                    for title in titles

                    if title.lower() == gold_leaf.lower()

                ),

                None

            )

            if matched_title is None:

                print(
                    f"Skipping line {line_number}: "
                    f"target section is not in ToC."
                )

                continue

            rows.append({

                "question": question,

                "gold_leaf": matched_title,

                "toc": titles

            })

    return rows


# --------------------------------------------------
# 4. Training and Validation Split
# --------------------------------------------------

def split_records(
    rows,
    validation_ratio=0.1
):

    if len(rows) < 10:

        return rows, []

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

        row

        for i, row in enumerate(rows)

        if i not in validation_indices

    ]

    validation_rows = [

        row

        for i, row in enumerate(rows)

        if i in validation_indices

    ]

    return train_rows, validation_rows


# --------------------------------------------------
# 5. Tokenized Dataset
# --------------------------------------------------

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

            completion = (
                " " + row["gold_leaf"]
            )

            # Tokenize prompt separately.

            prompt_ids = tokenizer(

                prompt,

                add_special_tokens=True,

                truncation=True,

                max_length=max_length - 80

            )["input_ids"]

            # Tokenize expected section title.

            completion_ids = tokenizer(

                completion + tokenizer.eos_token,

                add_special_tokens=False

            )["input_ids"]

            if len(completion_ids) >= max_length:
                continue

            # Prevent prompt and target from exceeding max length.

            available_length = (
                max_length - len(completion_ids)
            )

            prompt_ids = prompt_ids[
                :available_length
            ]

            input_ids = (
                prompt_ids + completion_ids
            )

            attention_mask = [
                1
            ] * len(input_ids)

            # Ignore prompt tokens in the loss.

            labels = (

                [-100] * len(prompt_ids)

                + completion_ids

            )

            self.items.append({

                "input_ids": input_ids,

                "attention_mask": attention_mask,

                "labels": labels

            })

    def __len__(self):

        return len(self.items)

    def __getitem__(self, index):

        return self.items[index]


# --------------------------------------------------
# 6. Main Training Function
# --------------------------------------------------

def main(

    train_file="data/stair_train.jsonl",

    output="artifacts/stair-finance-lora",

    epochs=2,

    max_length=2048,

    batch_size=1,

    learning_rate=2e-4

):

    # Check GPU availability.

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not found. "
            "Run this script using Google Colab GPU."
        )

    print("Loading training dataset...")

    rows = load_training_data(
        train_file
    )

    if len(rows) < 2:

        raise RuntimeError(
            "At least two valid training examples are required."
        )

    train_rows, validation_rows = split_records(
        rows
    )

    print(
        f"Total examples: {len(rows)}"
    )

    print(
        f"Training examples: {len(train_rows)}"
    )

    print(
        f"Validation examples: {len(validation_rows)}"
    )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Configure 4-bit quantization
    # --------------------------------------------------

    print("\nConfiguring QLoRA...")

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

    # --------------------------------------------------
    # Load Mistral model
    # --------------------------------------------------

    print("\nLoading Mistral 7B model...")

    model = AutoModelForCausalLM.from_pretrained(

        BASE_MODEL,

        quantization_config=quantization_config,

        device_map="auto"

    )

    model.config.use_cache = False

    # Prepare model for QLoRA.

    model = prepare_model_for_kbit_training(

        model,

        use_gradient_checkpointing=True

    )

    # --------------------------------------------------
    # Configure LoRA
    # --------------------------------------------------

    print("\nApplying LoRA configuration...")

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

    # --------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------

    print("\nPreparing datasets...")

    train_dataset = STAIRDataset(

        train_rows,

        tokenizer,

        max_length

    )

    validation_dataset = STAIRDataset(

        validation_rows,

        tokenizer,

        max_length

    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "No valid training examples were created."
        )

    if len(validation_dataset) == 0:

        validation_dataset = None

    print(
        f"Tokenized training samples: {len(train_dataset)}"
    )

    # --------------------------------------------------
    # Training arguments
    # --------------------------------------------------

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

    # Support different Transformers versions.

    training_signature = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if validation_dataset is not None:

        if "eval_strategy" in training_signature:

            training_parameters["eval_strategy"] = "epoch"

        elif "evaluation_strategy" in training_signature:

            training_parameters["evaluation_strategy"] = "epoch"

        training_parameters.update({

            "load_best_model_at_end": True,

            "metric_for_best_model": "eval_loss",

            "greater_is_better": False

        })

    training_args = TrainingArguments(
        **training_parameters
    )

    # --------------------------------------------------
    # Data collator
    # --------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(

        tokenizer=tokenizer,

        padding=True,

        label_pad_token_id=-100,

        return_tensors="pt"

    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        data_collator=data_collator

    )

    # --------------------------------------------------
    # Start training
    # --------------------------------------------------

    print("\nStarting Finance STAIR training...\n")

    trainer.train()

    # --------------------------------------------------
    # Save trained adapter
    # --------------------------------------------------

    print("\nSaving trained LoRA adapter...")

    trainer.save_model(
        str(output_path)
    )

    tokenizer.save_pretrained(
        str(output_path)
    )

    print("\nTraining completed.")

    print(
        f"Adapter saved at: {output_path}"
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    if validation_dataset is not None:

        print("\nEvaluating model...")

        metrics = trainer.evaluate()

        for key, value in metrics.items():

            print(
                f"{key}: {value}"
            )


# --------------------------------------------------
# 7. Command Line Arguments
# --------------------------------------------------

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
