import json, os, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from toc_generator import build_toc, attach_sections, toc_text

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

def predict_leaf(question,toc,adapter_path=None):
    # Uses trained LoRA adapter if present. Candidate constrained selection is used
    # as a safety check so the result must be one of the document's actual leaf titles.
    if adapter_path and Path(adapter_path).exists():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            base="mistralai/Mistral-7B-Instruct-v0.2"
            tok=AutoTokenizer.from_pretrained(adapter_path)
            model=AutoModelForCausalLM.from_pretrained(base,device_map="auto",
                torch_dtype=torch.float16,load_in_4bit=True)
            model=PeftModel.from_pretrained(model,adapter_path)
            prompt=("Select the single most relevant leaf section title from this ToC. "
                    "Output exactly one title.\nTOC:\n"+toc_text(toc)+
                    "\nQUESTION: "+question+"\nSECTION:")
            inputs=tok(prompt,return_tensors="pt",truncation=True,max_length=4096).to(model.device)
            out=model.generate(**inputs,max_new_tokens=64,do_sample=False)
            raw=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            for n in toc:
                if n["title"].lower() in raw.lower(): return n["title"]
        except Exception as e:
            print("Adapter inference unavailable; using candidate ranker:",e)
    # Fallback ranker is useful for local testing but is NOT the trained STAIR model.
    emb=load_embedder()
    candidates=[n["title"] for n in toc]
    q=emb.encode([question],normalize_embeddings=True)
    c=emb.encode(candidates,normalize_embeddings=True)
    return candidates[int(np.argmax(c @ q[0]))] if candidates else None

def retrieve_evidence(question, text, toc):
    sections = attach_sections(text, toc)

    # Also create page-based chunks as a fallback.
    pages = re.split(r"\[PAGE\s+(\d+)\]", text)

    page_chunks = []
    for i in range(1, len(pages), 2):
        page_no = pages[i]
        page_text = pages[i + 1].strip()

        if page_text:
            page_chunks.append({
                "title": f"PDF Page {page_no}",
                "text": page_text,
                "start_line": 0,
                "end_line": 0
            })

    candidates = sections + page_chunks

    emb = load_embedder()
    q = emb.encode([question], normalize_embeddings=True)

    docs = [
        item["title"] + "\n" + item["text"][:7000]
        for item in candidates
    ]

    d = emb.encode(docs, normalize_embeddings=True)
    scores = d @ q[0]

    order = np.argsort(scores)[::-1][:5]

    return [
        {**candidates[i], "score": float(scores[i])}
        for i in order
    ]

def answer_with_openrouter(question,evidence):
    import requests
    key=os.getenv("OPENROUTER_API_KEY","")
    if not key: return "Add OPENROUTER_API_KEY to Streamlit Secrets to enable answer generation."
    model=os.getenv("OPENROUTER_MODEL","openai/gpt-4o-mini")
    context="\n\n".join(f"SECTION: {e['title']}\n{e['text'][:5000]}" for e in evidence)
    prompt=("Answer only from the evidence below. If evidence is insufficient, say so. "
            "Do not invent financial figures. Cite section titles and page markers if present.\n\n"
            f"EVIDENCE:\n{context}\n\nQUESTION: {question}")
    r=requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        json={"model":model,"temperature":0.1,"messages":[{"role":"user","content":prompt}]},timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
