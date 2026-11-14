from pathlib import Path
from collections import Counter
root=Path("data/kaggle")
exts=Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
print("Files by extension:")
for ext,n in exts.most_common(): print(ext or "[no extension]",n)
print("\nSupported document types: PDF, TXT, MD, HTML, DOCX, CSV")
print("If the dataset is mostly numeric CSV, it is not appropriate for original ToC-based STAIR.")
