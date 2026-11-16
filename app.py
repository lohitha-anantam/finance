
import os
import tempfile
import streamlit as st
from pathlib import Path

from document_parser import read_file
from toc_generator import build_toc, toc_text
from stair_core import (
    predict_leaf,
    retrieve_evidence,
    answer_with_openrouter
)


# ==================================================
# 1. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Finance STAIR-RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Finance STAIR-RAG")

st.markdown(
    "Structure-Aware Retrieval-Augmented Generation "
    "for Financial Annual Reports"
)

st.divider()


# ==================================================
# 2. LOAD API CONFIGURATION
# ==================================================

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


# ==================================================
# 3. SESSION STATE
# ==================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}


# ==================================================
# 4. FILE PROCESSING FUNCTION
# ==================================================

def process_uploaded_file(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    temp_path = None

    try:

        # Save UploadedFile as a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())

            temp_path = Path(temp_file.name)

        # Pass Path object to document parser
        extracted_text = read_file(temp_path)

        if not extracted_text:
            raise ValueError(
                "No text was extracted from the document."
            )

        if not extracted_text.strip():
            raise ValueError(
                "The extracted document text is empty."
            )

        # Generate Table of Contents
        document_toc = build_toc(extracted_text)

        return extracted_text, document_toc

    finally:

        # Remove temporary file
        if temp_path and temp_path.exists():
            temp_path.unlink()


# ==================================================
# 5. SIDEBAR - UPLOAD DOCUMENTS
# ==================================================

st.sidebar.header("📁 Upload Annual Reports")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        if file_name not in st.session_state.documents:

            try:

                with st.spinner(
                    f"Processing {file_name}..."
                ):

                    extracted_text, document_toc = (
                        process_uploaded_file(uploaded_file)
                    )

                    st.session_state.documents[file_name] = {
                        "text": extracted_text,
                        "toc": document_toc
                    }

                st.sidebar.success(
                    f"Processed: {file_name}"
                )

            except Exception as error:

                st.sidebar.error(
                    f"Error processing {file_name}: {error}"
                )


# ==================================================
# 6. CHECK DOCUMENT AVAILABILITY
# ==================================================

if not st.session_state.documents:

    st.info(
        "Upload an annual report PDF from the sidebar to begin."
    )

    st.stop()


# ==================================================
# 7. SELECT DOCUMENT
# ==================================================

st.subheader("Select Annual Report")

document_names = list(
    st.session_state.documents.keys()
)

selected_document = st.selectbox(
    "Choose a document",
    document_names
)

document = st.session_state.documents[selected_document]

text = document["text"]

toc = document["toc"]


# ==================================================
# 8. DOCUMENT DIAGNOSTICS
# ==================================================

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

    st.markdown("### Detected Table of Contents")

    if toc:

        st.text(
            toc_text(toc)[:12000]
        )

    else:

        st.warning(
            "No headings detected. Check toc_generator.py."
        )

    if "[PAGE " not in text:

        st.warning(
            "Page markers are missing. "
            "Check whether document_parser.py preserves PDF page numbers."
        )


# ==================================================
# 9. QUESTION INPUT
# ==================================================

st.divider()

st.header("🔍 Ask Questions About the Annual Report")

question = st.text_area(
    "Enter your financial question:",
    placeholder=(
        "Example: How much did L&T spend on "
        "environmental management in FY 2024-25?"
    ),
    height=100
)


top_k = st.slider(
    "Number of evidence chunks to retrieve",
    min_value=3,
    max_value=10,
    value=5
)


# ==================================================
# 10. PROCESS QUESTION
# ==================================================

if st.button(
    "Generate Answer",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ------------------------------------------
        # A. PREDICT RELEVANT TOC SECTION
        # ------------------------------------------

        st.subheader("1. Predicted ToC Section")

        with st.spinner(
            "Predicting the relevant section..."
        ):

            try:

                predicted_leaf = predict_leaf(
                    question,
                    toc
                )

                if predicted_leaf:

                    st.success(predicted_leaf)

                else:

                    st.warning(
                        "No relevant ToC section predicted."
                    )

            except Exception as error:

                predicted_leaf = None

                st.error(
                    f"ToC prediction error: {error}"
                )


        # ------------------------------------------
        # B. HYBRID RETRIEVAL
        # ------------------------------------------

        st.subheader("2. Retrieved Evidence")

        with st.spinner(
            "Searching document using hybrid retrieval..."
        ):

            try:

                evidence = retrieve_evidence(
                    question=question,
                    text=text,
                    toc=toc,
                    predicted_leaf=predicted_leaf,
                    top_k=top_k
                )

            except Exception as error:

                evidence = []

                st.error(
                    f"Retrieval error: {error}"
                )


        # ------------------------------------------
        # C. DISPLAY RETRIEVED CHUNKS
        # ------------------------------------------

        if evidence:

            for index, item in enumerate(
                evidence,
                start=1
            ):

                title = item.get(
                    "title",
                    "Unknown Section"
                )

                score = item.get(
                    "score",
                    0.0
                )

                page = item.get("page")

                with st.expander(
                    f"Evidence {index}: {title} | Score: {score:.4f}",
                    expanded=(index == 1)
                ):

                    st.write(
                        "**Source:**",
                        item.get("source", "Unknown")
                    )

                    st.write(
                        "**PDF Page:**",
                        page if page is not None else "Unavailable"
                    )

                    st.markdown("**Retrieved Text:**")

                    st.text(
                        item.get("text", "")[:5000]
                    )

        else:

            st.error(
                "No relevant evidence was retrieved. "
                "Check document extraction and retrieval."
            )


        # ------------------------------------------
        # D. GENERATE FINAL ANSWER
        # ------------------------------------------

        st.subheader("3. Generated Answer")

        if evidence:

            if not api_key:

                st.error(
                    "OPENROUTER_API_KEY is missing. "
                    "Add it in Streamlit Cloud Secrets."
                )

            else:

                with st.spinner(
                    "Generating answer using OpenRouter..."
                ):

                    try:

                        answer = answer_with_openrouter(
                            question,
                            evidence
                        )

                        st.markdown(answer)

                    except Exception as error:

                        st.error(
                            f"Answer generation failed: {error}"
                        )

        else:

            st.warning(
                "Answer generation skipped because "
                "no relevant evidence was retrieved."
            )


        # ------------------------------------------
        # E. RETRIEVAL DIAGNOSTICS
        # ------------------------------------------

        st.subheader("4. Retrieval Diagnostics")

        if evidence:

            best = evidence[0]

            col1, col2 = st.columns(2)

            col1.metric(
                "Best Matching Score",
                f"{best.get('score', 0):.4f}"
            )

            col2.metric(
                "Retrieved Chunks",
                len(evidence)
            )

            st.write(
                "**Best Matching Section:**",
                best.get("title", "Unknown")
            )

            st.write(
                "**Best Matching PDF Page:**",
                best.get("page") or "Unavailable"
            )


# ==================================================
# 11. SIDEBAR DOCUMENT MANAGEMENT
# ==================================================

st.sidebar.divider()

st.sidebar.subheader("Document Management")

if st.sidebar.button("Clear Uploaded Documents"):

    st.session_state.documents = {}

    st.rerun()


# ==================================================
# 12. FOOTER
# ==================================================

st.divider()

st.caption(
    "Finance STAIR-RAG | Structure-Aware Financial "
    "Document Question Answering"
)
