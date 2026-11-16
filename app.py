
import os
import streamlit as st
from pathlib import Path

from document_parser import read_file
from toc_generator import build_toc, toc_text
from stair_core import (
    predict_leaf,
    retrieve_evidence,
    answer_with_openrouter
)

# --------------------------------------------------
# 1. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Finance STAIR-RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Finance STAIR-RAG")
st.caption(
    "Structure-Aware Retrieval-Augmented Generation "
    "for Financial Annual Reports"
)

# --------------------------------------------------
# 2. Load API configuration
# --------------------------------------------------

def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

api_key = get_secret("OPENROUTER_API_KEY")
model_name = get_secret(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)

if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

os.environ["OPENROUTER_MODEL"] = model_name

# --------------------------------------------------
# 3. Session state
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = {}

# --------------------------------------------------
# 4. Upload PDFs
# --------------------------------------------------

st.sidebar.header("Upload Annual Reports")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        if uploaded_file.name not in st.session_state.documents:

            try:
                text = read_file(uploaded_file)

                if not text or not text.strip():
                    st.error(
                        f"No text extracted from {uploaded_file.name}"
                    )
                    continue

                toc = build_toc(text)

                st.session_state.documents[
                    uploaded_file.name
                ] = {
                    "text": text,
                    "toc": toc
                }

                st.success(
                    f"Processed: {uploaded_file.name}"
                )

            except Exception as e:
                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

# --------------------------------------------------
# 5. Select document
# --------------------------------------------------

if not st.session_state.documents:

    st.info("Upload an annual report PDF to begin.")
    st.stop()

document_names = list(
    st.session_state.documents.keys()
)

selected_document = st.selectbox(
    "Select Annual Report",
    document_names
)

document = st.session_state.documents[
    selected_document
]

text = document["text"]
toc = document["toc"]

# --------------------------------------------------
# 6. Diagnostics
# --------------------------------------------------

with st.expander("📋 Document Diagnostics"):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Extracted Characters",
        f"{len(text):,}"
    )

    col2.metric(
        "Detected Headings",
        len(toc)
    )

    col3.metric(
        "Page Markers",
        text.count("[PAGE ")
    )

    st.write("Detected Table of Contents:")

    st.code(
        toc_text(toc)[:12000]
        if toc else "No headings detected."
    )

    if not toc:
        st.warning(
            "No headings detected. Check toc_generator.py."
        )

# --------------------------------------------------
# 7. Ask question
# --------------------------------------------------

st.divider()

st.subheader("Ask a Financial Question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What was the expenditure on "
        "environmental management in FY 2024-25?"
    ),
    height=100
)

top_k = st.slider(
    "Number of evidence chunks",
    min_value=3,
    max_value=10,
    value=5
)

# --------------------------------------------------
# 8. Process question
# --------------------------------------------------

if st.button(
    "Generate Answer",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Predicting relevant ToC section..."):

            predicted_leaf = predict_leaf(
                question,
                toc
            )

        st.subheader("1. Predicted ToC Section")

        st.write(
            predicted_leaf
            if predicted_leaf
            else "No relevant section predicted."
        )

        # ------------------------------------------
        # Hybrid retrieval
        # ------------------------------------------

        with st.spinner(
            "Searching document using hybrid retrieval..."
        ):

            evidence = retrieve_evidence(
                question=question,
                text=text,
                toc=toc,
                predicted_leaf=predicted_leaf,
                top_k=top_k
            )

        st.subheader("2. Retrieved Evidence")

        if not evidence:

            st.warning(
                "No evidence found. Check PDF extraction "
                "and section detection."
            )

        else:

            for index, item in enumerate(evidence, start=1):

                with st.expander(
                    f"Evidence {index}: {item['title']} "
                    f"(Score: {item['score']:.4f})",
                    expanded=(index == 1)
                ):

                    st.write(
                        "**Source:**",
                        item.get("source", "Unknown")
                    )

                    page = item.get("page")

                    st.write(
                        "**PDF Page:**",
                        page if page else "Not available"
                    )

                    st.write("**Retrieved Text:**")

                    st.text(
                        item["text"][:5000]
                    )

        # ------------------------------------------
        # Generate answer
        # ------------------------------------------

        st.subheader("3. Generated Answer")

        if evidence:

            with st.spinner(
                "Generating evidence-based answer..."
            ):

                try:

                    answer = answer_with_openrouter(
                        question,
                        evidence
                    )

                    st.success(answer)

                except Exception as e:

                    st.error(
                        f"Answer generation failed: {e}"
                    )

        else:

            st.warning(
                "Answer generation skipped because "
                "no evidence was retrieved."
            )

        # ------------------------------------------
        # Retrieval debugging
        # ------------------------------------------

        st.subheader("4. Retrieval Diagnostics")

        if evidence:

            best = evidence[0]

            st.write(
                "**Best matching section:**",
                best["title"]
            )

            st.write(
                "**Best matching score:**",
                round(best["score"], 4)
            )

            st.write(
                "**Best matching page:**",
                best.get("page") or "Unavailable"
            )

            if best.get("page") == "5":
                st.info(
                    "Check whether this page contains "
                    "the expected environmental expenditure."
                )

        st.caption(
            "Answers are generated only from retrieved evidence. "
            "Always verify important financial figures "
            "against the original report."
        )

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Finance STAIR-RAG | Evidence-based financial "
    "document question answering"
)
