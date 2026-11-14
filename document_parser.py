from pathlib import Path
import re, json, csv
import fitz
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm", ".docx", ".csv"}

def clean(s):
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n\n".join(f"[PAGE {i+1}]\n{p.get_text('text')}" for i,p in enumerate(doc))
    if ext in {".txt", ".md"}:
        return path.read_text(errors="ignore")
    if ext in {".html", ".htm"}:
        return BeautifulSoup(path.read_text(errors="ignore"), "html.parser").get_text("\n")
    if ext == ".docx":
        d = Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    if ext == ".csv":
        df = pd.read_csv(path).fillna("")
        # Convert each row to a document-like record; choose text-rich columns.
        rows=[]
        for i,row in df.iterrows():
            rows.append("\n".join(f"{c}: {row[c]}" for c in df.columns if str(row[c]).strip()))
        return "\n\n".join(rows)
    return ""

def iter_documents(root):
    root=Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            try:
                text=clean(read_file(p))
                if len(text) > 500:
                    yield {"doc_id": p.stem, "source": str(p), "text": text}
            except Exception as e:
                print(f"Skipping {p}: {e}")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kaggle")
    ap.add_argument("--out", default="data/documents.jsonl")
    a=ap.parse_args()
    docs=list(iter_documents(a.input))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w",encoding="utf8") as f:
        for d in docs: f.write(json.dumps(d,ensure_ascii=False)+"\n")
    print(f"Extracted {len(docs)} documents -> {a.out}")
