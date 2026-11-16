
"""
Finance STAIR - Streamlit Application

Workflow:
1. Upload financial report.
2. Extract document text.
3. Generate Table of Contents.
4. Predict relevant section using STAIR.
5. Retrieve relevant evidence.
6. Generate answer using OpenRouter.
7. Display supporting evidence.
"""

import os
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

from document_parser import read_file

from toc_generator import (
    build_toc,
    attach_sections,
    toc_text
)

from stair_core import (
    predict_leaf,
    retrieve_evidence,
    answer_with_openrouter
)


# --------------------------------------------------
# 1. Application Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(

    page_title="Finance STAIR",

    page_icon="📊",

    layout="wide"

)

st.title(
    "📊 Finance STAIR – Structure-Aware Financial Report Assistant"
)

st.caption(
    "Upload financial annual reports and ask questions "
    "using structure-aware retrieval and evidence-based generation."
)


# --------------------------------------------------
# 2. Load Streamlit Secrets
# --------------------------------------------------

try:

    for key in [

        "OPENROUTER_API_KEY",

        "OPENROUTER_MODEL",

        "STAIR_ADAPTER_PATH"

    ]:

        if key in st.secrets:

            os.environ[key] = str(
                st.secrets[key]
            )

except Exception:

    pass


# --------------------------------------------------
# 3. Upload Financial Report
# --------------------------------------------------

uploaded = st.file_uploader(

    "Upload a financial annual report",

    type=[
        "pdf",
        "txt",
        "md",
        "html",
        "htm",
        "docx"
    ]

)


if uploaded:

    # Save uploaded document

    data_dir = Path("data")

    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = data_dir / Path(
        uploaded.name
    ).name

    file_path.write_bytes(
        uploaded.getvalue()
    )

    # --------------------------------------------------
    # 4. Extract Document Text
    # --------------------------------------------------

    with st.spinner(
        "Extracting financial report text..."
    ):

        text = read_file(file_path)

    if not text or len(text.strip()) < 300:

        st.error(
            "Unable to extract sufficient text. "
            "The PDF may be scanned and require OCR."
        )

        st.stop()

    st.success(
        f"Successfully extracted {len(text):,} characters."
    )

    # --------------------------------------------------
    # 5. Generate Table of Contents
    # --------------------------------------------------

    with st.spinner(
        "Identifying report sections..."
    ):

        toc = build_toc(text)

        sections = attach_sections(
            text,
            toc
        )

    st.subheader(
        "Document Structure"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Extracted Characters",
        f"{len(text):,}"
    )

    col2.metric(
        "ToC Headings",
        len(toc)
    )

    col3.metric(
        "Extracted Sections",
        len(sections)
    )

    # --------------------------------------------------
    # 6. Display Table of Contents
    # --------------------------------------------------

    with st.expander(
        "View Extracted Table of Contents"
    ):

        if toc:

            st.code(
                toc_text(toc)
            )

        else:

            st.warning(
                "No headings were detected."
            )

        st.caption(
            "Verify these headings against the original annual report."
        )

    # --------------------------------------------------
    # 7. Debug Information
    # --------------------------------------------------

    with st.expander(
        "Document Extraction Diagnostics"
    ):

        page_markers = text.count("[PAGE ")

        st.write({

            "Document name": uploaded.name,

            "Extracted characters": len(text),

            "Detected ToC headings": len(toc),

            "Attached sections": len(sections),

            "Page markers": page_markers

        })

        st.write(
            "First 15 detected headings:"
        )

        st.write([
            item["title"]
            for item in toc[:15]
        ])

        if not sections:

            st.warning(
                "No structured sections were attached. "
                "The retrieval system will attempt text chunk fallback."
            )

    # --------------------------------------------------
    # 8. Ask Financial Question
    # --------------------------------------------------

    st.subheader(
        "Ask a Question"
    )

    question = st.text_area(

        "Enter your question about the financial report",

        placeholder=(
            "Example: What was the consolidated group turnover "
            "in FY 2024-25?"
        ),

        height=100

    )

    # --------------------------------------------------
    # 9. Process Question
    # --------------------------------------------------

    if st.button(

        "Find Section and Generate Answer",

        type="primary",

        disabled=not question.strip()

    ):

        adapter_path = os.getenv(

            "STAIR_ADAPTER_PATH",

            "artifacts/stair-finance-lora"

        )

        # ==============================================
        # Stage 1: Predict Relevant Section
        # ==============================================

        st.subheader(
            "1. Predicted ToC Section"
        )

        with st.spinner(
            "Predicting relevant financial section..."
        ):

            try:

                predicted_leaf = predict_leaf(

                    question,

                    toc,

                    adapter_path

                )

                st.write(
                    predicted_leaf or "No section identified."
                )

                if (
                    predicted_leaf
                    and not (
                        Path(adapter_path) / "adapter_config.json"
                    ).exists()
                ):

                    st.info(
                        "The trained LoRA adapter was not found. "
                        "The embedding-based baseline was used instead."
                    )

            except Exception as error:

                st.error(
                    f"Section prediction failed: {error}"
                )

                st.stop()

        # ==============================================
        # Stage 2: Retrieve Evidence
        # ==============================================

        st.subheader(
            "2. Retrieved Evidence"
        )

        with st.spinner(
            "Searching relevant financial report sections..."
        ):

            try:

                evidence = retrieve_evidence(

                    question=question,

                    text=text,

                    toc=toc,

                    predicted_leaf=predicted_leaf,

                    top_k=5

                )

            except Exception as error:

                st.error(
                    f"Evidence retrieval failed: {error}"
                )

                st.stop()

        if not evidence:

            st.warning(
                "No relevant evidence was retrieved. "
                "Check the extracted document text and section structure."
            )

        else:

            st.success(
                f"Retrieved {len(evidence)} evidence chunks."
            )

            for index, item in enumerate(
                evidence,
                1
            ):

                title = item.get(
                    "title",
                    "Unknown section"
                )

                score = item.get(
                    "score",
                    0.0
                )

                with st.expander(

                    f"{index}. {title} | Relevance: {score:.3f}",

                    expanded=(index == 1)

                ):

                    st.write(
                        item.get("text", "")[:10000]
                    )

                    if item.get("page"):

                        st.caption(
                            f"PDF page: {item['page']}"
                        )

                    elif item.get("page_start"):

                        st.caption(
                            f"Section starting around PDF page: "
                            f"{item['page_start']}"
                        )

                    elif item.get("start_line"):

                        st.caption(

                            f"Extracted lines: "
                            f"{item['start_line']} - "
                            f"{item.get('end_line')}"

                        )

                    else:

                        st.caption(
                            "Exact PDF page reference unavailable."
                        )

        # ==============================================
        # Stage 3: Generate Answer
        # ==============================================

        st.subheader(
            "3. Generated Answer"
        )

        if evidence:

            with st.spinner(
                "Generating evidence-based answer..."
            ):

                try:

                    answer = answer_with_openrouter(

                        question,

                        evidence

                    )

                    st.write(answer)

                except Exception as error:

                    st.error(
                        f"Answer generation failed: {error}"
                    )

        else:

            st.info(
                "Answer generation skipped because "
                "no supporting evidence was retrieved."
            )

        # ==============================================
        # Stage 4: Summary
        # ==============================================

        st.subheader(
            "4. Retrieval Summary"
        )

        st.write({

            "Document": uploaded.name,

            "Predicted section": predicted_leaf,

            "Retrieved evidence chunks": len(evidence),

            "Answer generated": bool(evidence)

        })


# --------------------------------------------------
# 10. Application Instructions
# --------------------------------------------------

else:

    st.info(
        "Upload a financial annual report to begin."
    )

    st.markdown(
        """
        **Supported file formats**

        - PDF
        - TXT
        - Markdown
        - HTML
        - DOCX

        **Example questions**

        - What was the company's revenue?
        - What was the consolidated turnover?
        - What was the net profit?
        - What are the major financial risks?
        - What were the total assets?
        - What was the dividend declared?
        """
    )
