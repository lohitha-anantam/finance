
"""
Finance STAIR Core
- Predicts a relevant leaf section using a trained LoRA adapter.
- Uses an embedding baseline if the adapter is unavailable.
- Retrieves evidence from the predicted section first.
- Performs global retrieval as a fallback.
- Generates answers using only retrieved evidence.
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


# ---------------------------------------------------------
# 1. ToC utilities
# ---------------------------------------------------------

def flatten_toc(toc):
    """
    Convert a nested or flat ToC into a list of leaf section titles.

    Supports common structures:
    {"title": "...", "children": [...]}
    {"title": "...", "subsections": [...]}
    {"title": "...", "sections": [...]}
    """

    leaves = []

    def visit(node):
        if isinstance(node, str):
            leaves.append(node.strip())
            return

        if not isinstance(node, dict):
            return

        title = str(node.get("title", "")).strip()

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
            leaves.append(title)

    for item in toc or []:
        visit(item)

    # Remove duplicate titles while preserving order
    result = []
    seen = set()

    for title in leaves:
        key = title.lower()

        if title and key not in seen:
            seen.add(key)
            result.append(title)

    return result


def format_toc_for_prompt(toc):
    """
    Keep this format identical in train_stair.py
    and generate_training_data.py.
    """

    titles = flatten_toc(toc)

    return "\n".join(f"- {title}" for title in titles)


def build_stair_prompt(question, toc):
    """
    Exact inference prompt used by the training script.
    """

    return (
        "You are a structure-aware financial document retriever.\n"
        "Given a financial question and the document table of contents, "
        "select the single most relevant leaf section.\n"
        "Output ONLY one exact section title from the supplied ToC.\n\n"
        f"TABLE OF CONTENTS:\n{format_toc_for_prompt(toc)}\n\n"
        f"QUESTION: {question}\n"
        "SECTION:"
    )


# ---------------------------------------------------------
# 2. Embedding model
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------
# 3. Load trained STAIR model
# ---------------------------------------------------------

@lru_cache(maxsize=2)
def load_stair_model(adapter_path):
    """
    Load the base Mistral model and its trained LoRA adapter.
    Cached so Streamlit does not reload the model for every question.
    """

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    from peft import PeftModel

    adapter = Path(adapter_path)

    if not adapter.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}"
        )

    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            "The adapter folder does not contain adapter_config.json. "
            "Train the model first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# 4. Predict relevant ToC leaf
# ---------------------------------------------------------

def predict_leaf(question, toc, adapter_path=None):

    candidates = flatten_toc(toc)

    if not candidates:
        return None

    # Use trained LoRA adapter if available
    if adapter_path and Path(adapter_path).exists():

        try:
            import torch

            tokenizer, model = load_stair_model(
                str(adapter_path)
            )

            prompt = build_stair_prompt(question, toc)

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

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = output[
                0, inputs["input_ids"].shape[1]:
            ]

            raw = tokenizer.decode(
                generated,
                skip_special_tokens=True
            ).strip()

            # Accept only an actual ToC leaf title
            raw_normalized = re.sub(
                r"\s+",
                " ",
                raw.lower()
            )

            for candidate in candidates:
                candidate_normalized = re.sub(
                    r"\s+",
                    " ",
                    candidate.lower()
                )

                if (
                    raw_normalized == candidate_normalized
                    or candidate_normalized in raw_normalized
                ):
                    return candidate

            print(
                "Model output did not match an exact ToC title:",
                raw
            )

        except Exception as error:
            print(
                "Trained adapter unavailable. "
                "Using embedding baseline:",
                error
            )

    # Fallback: embedding-based section ranking
    # This is NOT the trained STAIR model.
    try:
        embedder = load_embedder()

        question_embedding = embedder.encode(
            [question],
            normalize_embeddings=True
        )

        candidate_embeddings = embedder.encode(
            candidates,
            normalize_embeddings=True
        )

        scores = candidate_embeddings @ question_embedding[0]

        return candidates[int(np.argmax(scores))]

    except Exception as error:
        print("Embedding prediction failed:", error)
        return candidates[0]


# ---------------------------------------------------------
# 5. Financial metric matching
# ---------------------------------------------------------

def metric_terms(question):
    """
    Extract important financial concepts from a question.
    This helps distinguish similar financial metrics.
    """

    q = question.lower()

    groups = {
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
        "operating_profit": [
            "ebitda",
            "operating profit",
            "operating margin"
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
            "operating cash flow",
            "cash generated"
        ],
        "dividend": [
            "dividend",
            "dividend per share"
        ],
        "employees": [
            "employees",
            "workforce",
            "employee count"
        ]
    }

    matched = []

    for category, terms in groups.items():
        if any(term in q for term in terms):
            matched.append(category)

    return matched


def financial_metric_bonus(question, text):
    """
    Small relevance adjustment for financial terminology.
    """

    terms = metric_terms(question)

    if not terms:
        return 0.0

    content = text.lower()
    bonus = 0.0

    keyword_map = {
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
        "operating_profit": [
            "ebitda",
            "operating profit",
            "operating margin"
        ],
        "assets": ["total assets", "assets"],
        "debt": ["debt", "borrowings", "loans"],
        "cash_flow": ["cash flow", "cash generated"],
        "dividend": ["dividend"],
        "employees": ["employees", "workforce"]
    }

    for category in terms:
        if any(word in content for word in keyword_map[category]):
            bonus += 0.025

    return min(bonus, 0.10)


# ---------------------------------------------------------
# 6. Evidence retrieval
# ---------------------------------------------------------

def retrieve_evidence(
    question,
    text,
    toc,
    predicted_leaf=None,
    top_k=5
):
    """
    Retrieve evidence from the predicted section first.

    Also searches the remaining sections and PDF page chunks.
    This avoids losing the answer when the predicted section is wrong.
    """

    sections = attach_sections(text, toc)

    candidates = []

    # Add structured sections
    for section in sections:

        if not isinstance(section, dict):
            continue

        section_text = str(section.get("text", "")).strip()

        if not section_text:
            continue

        item = dict(section)

        item["title"] = str(
            section.get("title", "Untitled section")
        )

        item["text"] = section_text

        candidates.append(item)

    # Add page-based chunks for fallback retrieval
    page_pattern = r"\[PAGE\s+(\d+)\]"

    pages = re.split(page_pattern, text)

    for i in range(1, len(pages), 2):

        page_number = pages[i]
        page_text = pages[i + 1].strip()

        if not page_text:
            continue

        candidates.append({
            "title": f"PDF Page {page_number}",
            "text": page_text,
            "page": int(page_number),
            "start_line": None,
            "end_line": None
        })

    if not candidates:
        return []

    embedder = load_embedder()

    question_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    document_texts = [
        item["title"] + "\n" + item["text"][:7000]
        for item in candidates
    ]

    document_embeddings = embedder.encode(
        document_texts,
        normalize_embeddings=True
    )

    scores = document_embeddings @ question_embedding[0]

    predicted_normalized = (
        re.sub(r"\s+", " ", predicted_leaf.lower()).strip()
        if predicted_leaf
        else None
    )

    # Score adjustment
    for index, item in enumerate(candidates):

        score = float(scores[index])

        title_normalized = re.sub(
            r"\s+",
            " ",
            item["title"].lower()
        ).strip()

        # Prioritize the predicted leaf section
        if (
            predicted_normalized
            and title_normalized == predicted_normalized
        ):
            score += 0.12

        # Add small bonus for matching financial metrics
        score += financial_metric_bonus(
            question,
            item["title"] + "\n" + item["text"][:5000]
        )

        item["score"] = score

    # Sort by final relevance score
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates[:top_k]


# ---------------------------------------------------------
# 7. Evidence-based answer generation
# ---------------------------------------------------------

def answer_with_openrouter(question, evidence):

    import requests

    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "Add OPENROUTER_API_KEY to Streamlit Secrets."
        )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    if not evidence:
        return "No supporting evidence was retrieved."

    context_parts = []

    for item in evidence:

        title = item.get("title", "Unknown section")
        content = item.get("text", "")

        page = item.get("page")

        source = (
            f"PDF page {page}"
            if page is not None
            else "Page number unavailable"
        )

        context_parts.append(
            f"SECTION: {title}\n"
            f"SOURCE: {source}\n"
            f"EVIDENCE:\n{content[:5000]}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a financial annual-report analysis assistant.

Rules:
1. Answer only using the supplied evidence.
2. Never invent financial figures, percentages, dates or facts.
3. Distinguish consolidated figures from standalone figures.
4. Distinguish revenue, turnover, total income, profit and cash flow.
5. Pay attention to the financial year asked in the question.
6. Do not treat similarly named financial metrics as identical.
7. If the evidence does not contain the answer, clearly say:
   "The retrieved evidence does not provide enough information."
8. Include the relevant section title and PDF page number when available.
9. If multiple figures appear, explain which figure answers the question
   and why.
"""

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED EVIDENCE:\n{context}\n\n"
                    "Provide an accurate answer with its source."
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
