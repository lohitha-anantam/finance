import os, streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from document_parser import read_file
from toc_generator import build_toc, attach_sections, toc_text
from stair_core import predict_leaf, retrieve_evidence, answer_with_openrouter

load_dotenv()
st.set_page_config(page_title="Finance STAIR",page_icon="📊",layout="wide")
st.title("📊 Finance STAIR – Structure-Aware Financial Report Assistant")
st.caption("Upload annual reports / 10-K documents, locate relevant sections, and answer with evidence.")

# Streamlit Cloud secrets -> environment variables
for k in ["OPENROUTER_API_KEY","OPENROUTER_MODEL","KAGGLE_API_TOKEN","KAGGLE_DATASET"]:
    if k in st.secrets:
        os.environ[k]=str(st.secrets[k])

uploaded=st.file_uploader("Upload a financial report",type=["pdf","txt","md","html","htm","docx"])
if uploaded:
    tmp=Path("data")/uploaded.name
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(uploaded.getvalue())
    text=read_file(tmp)
    if len(text)<500:
        st.error("Could not extract enough text. Scanned PDFs need OCR before STAIR can process them.")
        st.stop()
    toc=build_toc(text)
    sections=attach_sections(text,toc)
    st.success(f"Extracted {len(text):,} characters and identified {len(toc)} candidate ToC headings.")
    with st.expander("View extracted Table of Contents"):
        st.code(toc_text(toc))
        st.caption("Headings are automatically inferred from report formatting and filing labels; verify them against the original report.")
    question=st.text_area("Ask a question about this financial report",placeholder="Example: What are the major risk factors disclosed by the company?")
    if st.button("Find section and generate answer",type="primary",disabled=not question.strip()):
        adapter=os.getenv("STAIR_ADAPTER_PATH","artifacts/stair-finance-lora")
        with st.spinner("Predicting the relevant section..."):
            leaf=predict_leaf(question,toc,adapter)
            evidence=retrieve_evidence(question,text,toc)
        st.subheader("Predicted ToC section")
        st.write(leaf or "No section identified")
        st.subheader("Answer")
        try:
            st.write(answer_with_openrouter(question,evidence))
        except Exception as e:
            st.error(f"Answer generation failed: {e}")
        st.subheader("Retrieved evidence")
        for i,e in enumerate(evidence,1):
            with st.expander(f"{i}. {e['title']} | relevance {e['score']:.3f}"):
                st.write(e["text"][:12000])
                st.caption(f"Source: {uploaded.name}; extracted line range {e['start_line']}–{e['end_line']}")
else:
    st.info("Upload a new annual report or 10-K PDF to test the system. Training corpus is prepared separately.")
