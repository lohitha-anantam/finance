"""LoRA fine-tuning: input=query + complete document ToC; target=gold leaf title."""
import json, argparse
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

def main(train_file="data/stair_train.jsonl", output="artifacts/stair-finance-lora", epochs=1):
    rows=[]
    for line in open(train_file,encoding="utf8"):
        x=json.loads(line)
        toc="\n".join(f"- {t}" for t in x["toc"])
        prompt=("You are a structure-aware financial document retriever. "
                "Given a question and the document table of contents, output ONLY the exact "
                "most relevant leaf section title from the ToC.\n"
                f"TABLE OF CONTENTS:\n{toc}\nQUESTION: {x['question']}\nSECTION:")
        rows.append({"text":prompt+" "+x["gold_leaf"]})
    if not rows: raise RuntimeError("No training examples found.")
    ds=Dataset.from_list(rows)
    tok=AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    quant=None
    model=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,device_map="auto",torch_dtype=torch.float16,
        load_in_4bit=True
    )
    peft=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                    bias="none",task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj"])
    args=TrainingArguments(output_dir=output, num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, logging_steps=5, save_strategy="epoch",
        fp16=True, report_to="none", gradient_checkpointing=True)
    trainer=SFTTrainer(model=model,args=args,train_dataset=ds,
        peft_config=peft,processing_class=tok,
        dataset_text_field="text",max_seq_length=4096)
    trainer.train()
    trainer.save_model(output); tok.save_pretrained(output)
    print("Saved LoRA adapter:",output)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--train-file",default="data/stair_train.jsonl"); p.add_argument("--output",default="artifacts/stair-finance-lora"); p.add_argument("--epochs",type=int,default=1); a=p.parse_args(); main(a.train_file,a.output,a.epochs)
