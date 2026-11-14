import os, json, requests, argparse
from pathlib import Path
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

SYSTEM = """You create high-quality retrieval training questions for financial annual reports.
Use only the supplied section text. Return strict JSON array. Each item must contain:
question (a natural question answerable from this section), answer (short evidence-based answer),
gold_leaf (exact section title). Make 2 questions. Do not invent facts."""
def ask(section, title):
    if not OPENROUTER_API_KEY: raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    payload={"model":OPENROUTER_MODEL,"temperature":0.2,
      "messages":[{"role":"system","content":SYSTEM},
      {"role":"user","content":f"SECTION TITLE: {title}\nSECTION TEXT:\n{section[:10000]}"}]}
    r=requests.post("https://openrouter.ai/api/v1/chat/completions",
       headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"},
       json=payload,timeout=120)
    r.raise_for_status()
    content=r.json()["choices"][0]["message"]["content"]
    content=content.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(content)

def main(src="data/structured.jsonl",out="data/stair_train.jsonl",limit=100):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    count=0
    with open(src,encoding="utf8") as inp, open(out,"w",encoding="utf8") as dst:
        for line in inp:
            rec=json.loads(line)
            for sec in rec["sections"]:
                try: items=ask(sec["text"],sec["title"])
                except Exception as e:
                    print("QA generation failed:",e); continue
                for x in items:
                    x.update({"doc_id":rec["doc_id"],"toc":[n["title"] for n in rec["toc"]]})
                    dst.write(json.dumps(x,ensure_ascii=False)+"\n"); count+=1
            if count>=limit: break
    print(f"Wrote {count} examples to {out}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=100); a=p.parse_args(); main(limit=a.limit)
