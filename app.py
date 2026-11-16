
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from document_parser import read_file
from toc_generator import build_toc, toc_text
from stair_core import (
    predict_leaf,
    retrieve_evidence,
    answer_with_openrouter
)

load_dotenv()

st.set_page_config(
    page_title="Finance STAIR",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Finance STAIR – Structure-Aware Financial Report Assistant")
st.caption(
    "Upload annual reports / 10-K documents, locate relevant sections, "
    "and generate answers using retrieved evidence."
)

# Load Streamlit Cloud secrets into environment variables
try:
    for key in [
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "KAGGLE_API_TOKEN",
        "KAGGLE_DATASET",
        "STAIR_ADAPTER_PATH"
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

# Upload financial report
uploaded = st.file_uploader(
    "Upload a financial report",
    type=["pdf", "txt", "md", "html", "htm", "docx"]
)

if uploaded:

    # Save uploaded document
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    tmp = data_dir / uploaded.name
    tmp.write_bytes(uploaded.getvalue())

    # Extract document text
    with st.spinner("Extracting document text..."):
        text = read_file(tmp)

    if len(text) < 500:
        st.error(
            "Could not extract enough text. "
            "Scanned PDFs need OCR before STAIR can process them."
        )
        st.stop()

    # Build hierarchical Table of Contents
    with st.spinner("Identifying document structure..."):
        toc = build_toc(text)

    st.success(
        f"Extracted {len(text):,} characters and "
        f"identified {len(toc)} candidate ToC headings."
    )

    # Display ToC
    with st.expander("View extracted Table of Contents"):
        st.code(toc_text(toc))
        st.caption(
            "Headings are automatically inferred from report formatting "
            "and filing labels. Verify them against the original report."
        )

    # User question
    question = st.text_area(
        "Ask a question about this financial report",
        placeholder=(
            "Example: What was the consolidated group turnover "
            "in FY 2024-25?"
        )
    )

    if st.button(
        "Find section and generate answer",
        type="primary",
        disabled=not question.strip()
    ):

        adapter = os.getenv(
            "STAIR_ADAPTER_PATH",
            "artifacts/stair-finance-lora"
        )

        # Stage 1: STAIR section prediction
        with st.spinner("Predicting the relevant section..."):
            try:
                leaf = predict_leaf(
                    question,
                    toc,
                    adapter
                )
            except Exception as e:
                st.error(f"Section prediction failed: {e}")
                st.stop()

        st.subheader("1. Predicted ToC section")
        st.write(leaf if leaf else "No section identified")

        # Stage 2: Evidence retrieval using predicted section
        with st.spinner("Retrieving relevant evidence..."):
            try:
                evidence = retrieve_evidence(
                    question=question,
                    text=text,
                    toc=toc,
                    predicted_leaf=leaf
                )
            except Exception as e:
                st.error(f"Evidence retrieval failed: {e}")
                st.stop()

        st.subheader("2. Retrieved evidence")

        if not evidence:
            st.warning("No relevant evidence was found.")
            st.stop()

        # Stage 3: Evidence-based answer generation
        st.subheader("3. Answer")

        try:
            answer = answer_with_openrouter(
                question,
                evidence
            )
            st.write(answer)

        except Exception as e:
            st.error(f"Answer generation failed: {e}")

        # Stage 4: Show supporting evidence
        st.subheader("4. Supporting evidence")

        for i, item in enumerate(evidence, 1):

            title = item.get("title", "Untitled section")
            score = item.get("score", 0.0)

            with st.expander(
                f"{i}. {title} | Relevance: {score:.3f}"
            ):

                st.write(item.get("text", "No text available." )[:12000])

                # Display available source location
                page = item.get("page")
                page_start = item.get("page_start")
                page_end = item.get("page_end")

                if page is not None:
                    st.caption(f"Source: {uploaded.name} | PDF page {page}")

                elif page_start is not None:
                    st.caption(
                        f"Source: {uploaded.name} | "
                        f"PDF pages {page_start}–{page_end or page_start}"
                    )

                else:
                    start = item.get("start_line")
                    end = item.get("end_line")

                    if start is not None and end is not None:
                        st.caption(
                            f"Source: {uploaded.name} | "
                            f"Extracted lines {start}–{end}"
                        )

else:
    st.info(
        "Upload a new annual report or 10-K PDF to test the system. "
        "Training corpus is prepared separately."
    )
