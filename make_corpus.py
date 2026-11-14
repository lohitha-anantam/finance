import json
from pathlib import Path
from document_parser import iter_documents
from toc_generator import build_toc, attach_sections

def main(input_dir="data/kaggle", out="data/structured.jsonl"):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    count=0
    with open(out,"w",encoding="utf8") as f:
        for d in iter_documents(input_dir):
            toc=build_toc(d["text"])
            sections=attach_sections(d["text"],toc)
            if not sections: continue
            rec={"doc_id":d["doc_id"],"source":d["source"],"text":d["text"],
                 "toc":toc,"sections":sections}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); count+=1
    print(f"Prepared {count} structured financial documents: {out}")
if __name__=="__main__": main()
