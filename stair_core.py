
"""
Finance STAIR Core

1. Predict relevant ToC section.
2. Retrieve evidence from the predicted section.
3. Search other sections when necessary.
4. Generate answers using OpenRouter.
"""

import os
import re
from pathlib import Path
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from toc_generator import attach_sections, toc_text


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 1. Flatten Table of Contents
# --------------------------------------------------

def flatten_toc(toc):

    output = []

    def visit(node):

        if isinstance(node, str):

            if node.strip():
                output.append(node.strip())

            return

        if not isinstance(node, dict):
            return

        title = str(
            node.get("title", "")
        ).strip()

        children = (
            node.get("children")
            or node.get("subsections")
            or node.get("sections")
            or []
        )

        if children:

            for child in children:
                visit(child)

        elif title:

            output.append(title)

    for node in toc or []:
        visit(node)

    result = []
    seen = set()

    for title in output:

        key = title.lower()

        if key not in seen:
            seen.add(key)
            result.append(title)

    return result


# --------------------------------------------------
# 2. Build STAIR Prompt
# --------------------------------------------------

def build_stair_prompt(question, toc):

    titles = flatten_toc(toc)

    formatted_toc = "\n".join(
        f"- {title}"
        for title in titles
    )

    return (
        "You are a structure-aware financial document retriever.\n"
        "Choose the single most relevant section title "
        "from the supplied list.\n"
        "Return exactly one title copied from the list. "
        "Do not return a sentence from the report.\n\n"
        f"TABLE OF CONTENTS:\n{formatted_toc}\n\n"
        f"QUESTION: {question}\n"
        "SECTION:"
    )


# --------------------------------------------------
# 3. Load Embedding Model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_embedder():

    return SentenceTransformer(
        EMBED_MODEL
    )


# --------------------------------------------------
# 4. Load Trained LoRA Model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_stair_model(adapter_path):

    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )

    from peft import PeftModel

    adapter = Path(adapter_path)

    if not (
        adapter / "adapter_config.json"
    ).exists():

        raise FileNotFoundError(
            "Trained STAIR adapter not found."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    model.eval()

    return tokenizer, model


# --------------------------------------------------
# 5. Predict Relevant Section
# --------------------------------------------------

def predict_leaf(question, toc, adapter_path=None):

    candidates = flatten_toc(toc)

    if not candidates:
        return None

    # Use trained STAIR adapter
    if (
        adapter_path
        and (
            Path(adapter_path) / "adapter_config.json"
        ).exists()
    ):

        try:

            import torch

            tokenizer, model = load_stair_model(
                str(adapter_path)
            )

            prompt = build_stair_prompt(
                question,
                toc
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500
            )

            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    max_new_tokens=48,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            normalized = re.sub(
                r"\s+",
                " ",
                generated
            ).strip().lower()

            # Only accept an exact ToC title
            for candidate in candidates:

                expected = re.sub(
                    r"\s+",
                    " ",
                    candidate
                ).strip().lower()

                if normalized == expected:

                    return candidate

            print(
                "Generated title does not match ToC:",
                generated
            )

        except Exception as error:

            print(
                "STAIR adapter failed. "
                "Using embedding fallback:",
                error
            )

    # Embedding fallback
    # This is not the trained STAIR model.

    embedder = load_embedder()

    question_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    candidate_embeddings = embedder.encode(
        candidates,
        normalize_embeddings=True
    )

    scores = (
        candidate_embeddings
        @ question_embedding[0]
    )

    return candidates[
        int(np.argmax(scores))
    ]


# --------------------------------------------------
# 6. Financial Metric Matching
# --------------------------------------------------

def metric_terms(question):

    q = question.lower()

    categories = {

        "revenue": [
            "revenue",
            "turnover",
            "sales",
            "income from operations"
        ],

        "profit": [
            "profit",
            "net profit",
            "pat",
            "profit after tax"
        ],

        "ebitda": [
            "ebitda",
            "operating profit"
        ],

        "assets": [
            "total assets",
            "assets"
        ],

        "debt": [
            "debt",
            "borrowings",
            "loans"
        ],

        "cash_flow": [
            "cash flow",
            "cash generated"
        ],

        "dividend": [
            "dividend"
        ],

        "employees": [
            "employees",
            "workforce"
        ]
    }

    matched = []

    for category, words in categories.items():

        if any(word in q for word in words):

            matched.append(category)

    return matched


def financial_metric_bonus(question, text):

    terms = metric_terms(question)

    content = text.lower()

    keywords = {

        "revenue": [
            "revenue",
            "turnover",
            "sales",
            "income from operations"
        ],

        "profit": [
            "profit",
            "pat",
            "profit after tax"
        ],

        "ebitda": [
            "ebitda",
            "operating profit"
        ],

        "assets": [
            "assets"
        ],

        "debt": [
            "debt",
            "borrowings"
        ],

        "cash_flow": [
            "cash flow"
        ],

        "dividend": [
            "dividend"
        ],

        "employees": [
            "employees",
            "workforce"
        ]
    }

    bonus = 0.0

    for category in terms:

        if any(
            word in content
            for word in keywords[category]
        ):

            bonus += 0.025

    return min(bonus, 0.10)


# --------------------------------------------------
# 7. PDF Page Chunk Extraction
# --------------------------------------------------

def create_page_chunks(text):

    pattern = r"\[PAGE\s+(\d+)\]"

    pages = re.split(pattern, text)

    chunks = []

    for i in range(1, len(pages), 2):

        page_number = int(pages[i])

        page_text = pages[i + 1].strip()

        if page_text:

            chunks.append({

                "title": f"PDF Page {page_number}",

                "text": page_text,

                "page": page_number,

                "start_line": None,

                "end_line": None

            })

    return chunks


# --------------------------------------------------
# 8. Evidence Retrieval
# --------------------------------------------------

def retrieve_evidence(
    question,
    text,
    toc,
    predicted_leaf=None,
    top_k=5
):

    # Extract structured sections

    sections = attach_sections(
        text,
        toc
    )

    candidates = [

        dict(section)

        for section in sections

        if section.get("text", "").strip()

    ]

    # Add page chunks

    page_chunks = create_page_chunks(text)

    candidates.extend(page_chunks)

    # Fallback when no sections or page markers exist

    if not candidates:

        lines = text.splitlines()

        chunk_size = 45

        overlap = 8

        step = chunk_size - overlap

        for start in range(
            0,
            len(lines),
            step
        ):

            end = min(
                start + chunk_size,
                len(lines)
            )

            content = "\n".join(
                lines[start:end]
            ).strip()

            if len(content) >= 100:

                candidates.append({

                    "title": (
                        f"Extracted lines {start + 1}-{end}"
                    ),

                    "text": content,

                    "start_line": start + 1,

                    "end_line": end

                })

    if not candidates:

        print("No retrieval candidates found.")

        return []

    print(
        "DEBUG: Total retrieval candidates:",
        len(candidates)
    )

    print(
        "DEBUG: Predicted section:",
        predicted_leaf
    )

    # Generate embeddings

    embedder = load_embedder()

    question_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    document_texts = [

        item["title"] + "\n" + item["text"][:6000]

        for item in candidates

    ]

    document_embeddings = embedder.encode(
        document_texts,
        normalize_embeddings=True
    )

    scores = (
        document_embeddings
        @ question_embedding[0]
    )

    # Give a small preference to predicted section

    target = (
        predicted_leaf.lower().strip()
        if predicted_leaf
        else None
    )

    for i, item in enumerate(candidates):

        score = float(scores[i])

        title = item["title"].lower().strip()

        if target and title == target:

            score += 0.08

        score += financial_metric_bonus(
            question,
            item["title"] + "\n" + item["text"][:5000]
        )

        item["score"] = score

    # Sort by relevance

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates[:max(1, top_k)]


# --------------------------------------------------
# 9. OpenRouter Answer Generation
# --------------------------------------------------

def answer_with_openrouter(question, evidence):

    import requests

    api_key = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    )

    if not api_key:

        raise RuntimeError(
            "Add OPENROUTER_API_KEY to Streamlit Secrets."
        )

    if not evidence:

        return (
            "No evidence was retrieved. "
            "Check document extraction and retrieval."
        )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    context = []

    for item in evidence:

        title = item.get(
            "title",
            "Unknown section"
        )

        page = item.get("page")

        source = (
            f"PDF page {page}"
            if page
            else title
        )

        content = item.get(
            "text",
            ""
        )

        context.append(

            f"SOURCE: {source}\n"
            f"SECTION: {title}\n"
            f"EVIDENCE:\n{content[:4500]}"

        )

    system_prompt = """
You are a financial annual-report analysis assistant.

Follow these rules:

1. Answer only from the supplied evidence.
2. Never invent financial figures.
3. Distinguish consolidated and standalone figures.
4. Distinguish turnover, revenue and total income.
5. Distinguish profit and cash flow.
6. Pay attention to the financial year.
7. If evidence is insufficient, say so.
8. Cite the section title and PDF page when available.
9. If multiple figures appear, explain which one answers
   the question and why.
"""

    payload = {

        "model": model,

        "temperature": 0,

        "messages": [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",

                "content": (

                    f"QUESTION:\n{question}\n\n"

                    "RETRIEVED EVIDENCE:\n"

                    + "\n\n---\n\n".join(context)

                )

            }

        ]

    }

    response = requests.post(

        "https://openrouter.ai/api/v1/chat/completions",

        headers={

            "Authorization": f"Bearer {api_key}",

            "Content-Type": "application/json"

        },

        json=payload,

        timeout=120

    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()
