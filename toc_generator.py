import re
from collections import OrderedDict

# Candidate headings common in annual reports and SEC 10-K filings.
HEADING_PATTERNS = [
 r"^(?:PART\s+[IVX]+[\.: -].*)$",
 r"^(?:ITEM\s+\d+[A-Z]?(?:[\.: -].*)?)$",
 r"^(?:[0-9]+(?:\.[0-9]+)*[\.)]?\s+[A-Z][A-Za-z0-9 ,/&()'’:-]{3,100})$",
 r"^(?:MANAGEMENT(?:'S)? DISCUSSION AND ANALYSIS.*)$",
 r"^(?:CONSOLIDATED (?:BALANCE SHEETS|STATEMENTS.*))$",
 r"^(?:RISK FACTORS.*|BUSINESS.*|LEGAL PROCEEDINGS.*|NOTES TO.*)$",
]
patterns=[re.compile(p,re.I) for p in HEADING_PATTERNS]

def build_toc(text, max_nodes=160):
    lines=[re.sub(r"\s+"," ",x).strip() for x in text.splitlines()]
    nodes=[]
    for idx,line in enumerate(lines):
        if not line or len(line)>150: continue
        if any(p.match(line) for p in patterns):
            # Avoid repeated page labels and overly generic false positives.
            if line.upper().startswith("[PAGE"): continue
            nodes.append({"title":line,"line_index":idx})
    # Deduplicate repeated headings while retaining first occurrence.
    seen=set(); out=[]
    for n in nodes:
        k=n["title"].lower()
        if k not in seen:
            seen.add(k); out.append(n)
    if not out:
        # Fallback: explicit inferred sections. Clearly marked as inferred, not official.
        out=[{"title":f"Inferred section {i+1}","line_index":int(i*len(lines)/min(12,max(1,len(lines))))}
             for i in range(min(12,max(1,len(lines)//120)))]
    return out[:max_nodes]

def attach_sections(text, toc):
    lines=text.splitlines()
    result=[]
    for i,node in enumerate(toc):
        start=node["line_index"]
        end=toc[i+1]["line_index"] if i+1<len(toc) else len(lines)
        body="\n".join(lines[start:end]).strip()
        if len(body)>100:
            result.append({"title":node["title"],"text":body,"start_line":start,"end_line":end})
    return result

def toc_text(toc):
    return "\n".join(f"- {n['title']}" for n in toc)
