
import os
import re
import numpy as np

from functools import lru_cache
from rank_bm25 import BM25Okapi

from toc_generator import attach_sections


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60

# Initial threshold only. Calibrate using labelled finance chunks.
RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "0.50")
)


# ============================================================
# 1. LOAD MODELS
# ============================================================

@lru_cache(maxsize=1)
def load_embedder():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def load_reranker():

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


# ============================================================
# 2. FLATTEN HIERARCHICAL TABLE OF CONTENTS
# ============================================================

def flatten_toc(toc):

    result = []

    def visit(nodes):

        for node in nodes or []:

            if isinstance(node, str):

                result.append(node)

            elif isinstance(node, dict):

                title = node.get("title", "")

                if title:
                    result.append(title)

                for key in (
                    "children",
                    "subsections",
                    "sections"
                ):

                    if isinstance(node.get(key), list):
                        visit(node[key])

    visit(toc)

    return list(dict.fromkeys(result))


# ============================================================
# 3. PREDICT RELEVANT TOC SECTION
# ============================================================

def predict_leaf(question, toc, adapter_path=None):

    candidates = flatten_toc(toc)

    if not candidates:
        return None

    try:

        model = load_embedder()

        embeddings = model.encode(
            [question] + candidates,
            normalize_embeddings=True
        )

        scores = embeddings[1:] @ embeddings[0]

        return candidates[int(np.argmax(scores))]

    except Exception as e:

        print("ToC prediction error:", e)

        return candidates[0]


# ============================================================
# 4. CREATE PAGE-BASED CHUNKS
# ============================================================

def create_page_chunks(
    text,
    chunk_size=1200,
    overlap=200
):

    pages = re.split(
        r"\[PAGE\s+(\d+)\]",
        text
    )

    chunks = []

    # PDF text with page markers

    if len(pages) > 1:

        for i in range(1, len(pages), 2):

            page_number = pages[i]

            page_text = pages[i + 1].strip()

            if not page_text:
                continue

            start = 0

            while start < len(page_text):

                end = start + chunk_size

                chunk_text = page_text[start:end].strip()

                if chunk_text:

                    chunks.append({

                        "title": f"PDF Page {page_number}",

                        "text": chunk_text,

                        "page": page_number

                    })

                start += chunk_size - overlap

    # Fallback without page markers

    else:

        start = 0

        page_number = 1

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({

                    "title": f"Text Chunk {page_number}",

                    "text": chunk_text,

                    "page": None

                })

            start += chunk_size - overlap

            page_number += 1

    return chunks


# ============================================================
# 5. FINANCIAL METRIC MATCHING
# ============================================================

def metric_bonus(question, document):

    q = question.lower()

    d = document.lower()

    metric_groups = {

        "energy": [
            "energy consumption",
            "energy consumed",
            "energy usage",
            "direct energy",
            "indirect energy"
        ],

        "environment": [
            "environmental management",
            "spend on environment",
            "pollution control",
            "waste management",
            "wastewater treatment"
        ],

        "csr": [
            "csr spend",
            "csr expenditure",
            "csr partners",
            "corporate social responsibility"
        ],

        "turnover": [
            "turnover",
            "revenue",
            "consolidated turnover"
        ],

        "order_book": [
            "order book",
            "order backlog"
        ],

        "employees": [
            "employees",
            "workers",
            "workforce"
        ],

        "emissions": [
            "ghg emissions",
            "emission intensity",
            "carbon emissions"
        ],

        "dividend": [
            "dividend",
            "dividend payout"
        ],

        "profit": [
            "profit",
            "pbit",
            "pat",
            "net profit"
        ],

        "water": [
            "water consumption",
            "water consumed",
            "water usage"
        ]

    }

    bonus = 0.0

    for terms in metric_groups.values():

        if any(term in q for term in terms):

            if any(term in d for term in terms):

                bonus += 0.10

    return min(bonus, 0.30)


# ============================================================
# 6. REMOVE DUPLICATE CANDIDATES
# ============================================================

def remove_duplicates(candidates):

    unique = {}

    for item in candidates:

        key = item["text"][:500].strip().lower()

        if key and key not in unique:

            unique[key] = item

    return list(unique.values())


# ============================================================
# 7. VECTOR SEARCH
# ============================================================

def vector_search(question, candidates):

    if not candidates:
        return []

    model = load_embedder()

    documents = [

        item["title"] + " " + item["text"]

        for item in candidates

    ]

    query_embedding = model.encode(

        [question],

        normalize_embeddings=True

    )

    document_embeddings = model.encode(

        documents,

        normalize_embeddings=True

    )

    scores = document_embeddings @ query_embedding[0]

    results = []

    for item, score in zip(candidates, scores):

        results.append({

            **item,

            "vector_score": float(score)

        })

    results.sort(

        key=lambda x: x["vector_score"],

        reverse=True

    )

    return results


# ============================================================
# 8. BM25 KEYWORD SEARCH
# ============================================================

def bm25_search(question, candidates):

    if not candidates:
        return []

    documents = [

        re.findall(
            r"[a-zA-Z0-9₹]+",
            item["title"] + " " + item["text"].lower()
        )

        for item in candidates

    ]

    tokenized_query = re.findall(

        r"[a-zA-Z0-9₹]+",

        question.lower()

    )

    bm25 = BM25Okapi(documents)

    scores = bm25.get_scores(tokenized_query)

    results = []

    for item, score in zip(candidates, scores):

        results.append({

            **item,

            "bm25_score": float(score)

        })

    results.sort(

        key=lambda x: x["bm25_score"],

        reverse=True

    )

    return results


# ============================================================
# 9. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(

    vector_results,

    keyword_results,

    k=RRF_K

):

    fused_scores = {}

    documents = {}

    # Vector ranking

    for rank, doc in enumerate(vector_results, start=1):

        doc_id = doc["id"]

        fused_scores[doc_id] = (

            fused_scores.get(doc_id, 0.0)

            + 1.0 / (k + rank)

        )

        documents[doc_id] = doc

    # BM25 ranking

    for rank, doc in enumerate(keyword_results, start=1):

        doc_id = doc["id"]

        fused_scores[doc_id] = (

            fused_scores.get(doc_id, 0.0)

            + 1.0 / (k + rank)

        )

        documents[doc_id] = {

            **documents.get(doc_id, {}),

            **doc

        }

    ranked = sorted(

        fused_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    results = []

    for doc_id, score in ranked:

        results.append({

            **documents[doc_id],

            "rrf_score": float(score)

        })

    return results


# ============================================================
# 10. CROSS-ENCODER RERANKING
# ============================================================

def cross_encoder_rerank(question, candidates):

    if not candidates:
        return []

    model = load_reranker()

    pairs = [

        (question, item["title"] + " " + item["text"])

        for item in candidates

    ]

    logits = model.predict(

        pairs,

        batch_size=16,

        show_progress_bar=False

    )

    # Sigmoid transforms logits to a 0-1 scale.
    # These are NOT calibrated probabilities.

    scores = 1.0 / (

        1.0 + np.exp(

            -np.clip(logits, -50, 50)

        )

    )

    results = []

    for item, score in zip(candidates, scores):

        results.append({

            **item,

            "relevance_score": float(score)

        })

    results.sort(

        key=lambda x: x["relevance_score"],

        reverse=True

    )

    return results


# ============================================================
# 11. COMPLETE HYBRID RETRIEVAL PIPELINE
# ============================================================

def retrieve_evidence(

    question,

    text,

    toc,

    predicted_leaf=None,

    top_k=5,

    threshold=None

):

    if threshold is None:

        threshold = RELEVANCE_THRESHOLD

    candidates = []

    # A. Extract ToC-based sections

    try:

        sections = attach_sections(text, toc)

        for section in sections:

            section_text = section.get("text", "")

            if section_text.strip():

                candidates.append({

                    "title": section.get(
                        "title",
                        "Document Section"
                    ),

                    "text": section_text,

                    "page": section.get("page"),

                    "source": "section"

                })

    except Exception as e:

        print("Section extraction error:", e)

    # B. Add page-based chunks

    page_chunks = create_page_chunks(text)

    for chunk in page_chunks:

        candidates.append({

            **chunk,

            "source": "page"

        })

    if not candidates:

        return []

    # C. Remove duplicate chunks

    candidates = remove_duplicates(candidates)

    # D. Add unique IDs

    for i, item in enumerate(candidates):

        item["id"] = str(i)

    # E. Apply STAIR section preference

    if predicted_leaf:

        matching = [

            item for item in candidates

            if predicted_leaf.lower() in item["title"].lower()

        ]

        # Keep all candidates if no section title matches.
        # Page chunks remain available as supporting evidence.

        if matching:

            matching_ids = {

                item["id"]

                for item in matching

            }

            candidates = [

                item for item in candidates

                if (

                    item["id"] in matching_ids

                    or item["source"] == "page"

                )

            ]

    # F. Vector search

    vector_results = vector_search(

        question,

        candidates

    )

    # G. BM25 search

    keyword_results = bm25_search(

        question,

        candidates

    )

    # H. Reciprocal Rank Fusion

    fused_results = reciprocal_rank_fusion(

        vector_results,

        keyword_results

    )

    # I. Cross-Encoder reranking

    reranked_results = cross_encoder_rerank(

        question,

        fused_results[:max(top_k * 5, 20)]

    )

    # J. Add financial metric matching

    final_results = []

    for item in reranked_results:

        financial_score = metric_bonus(

            question,

            item["title"] + " " + item["text"]

        )

        item["financial_score"] = float(financial_score)

        item["is_relevant"] = (

            item["relevance_score"] >= threshold

        )

        final_results.append(item)

    # K. Return top results, preserving both categories

    return final_results[:top_k]


# ============================================================
# 12. GET RELEVANT AND IRRELEVANT CHUNKS
# ============================================================

def split_retrieval_results(results):

    relevant = [

        item for item in results

        if item.get("is_relevant", False)

    ]

    irrelevant = [

        item for item in results

        if not item.get("is_relevant", False)

    ]

    return relevant, irrelevant


# ============================================================
# 13. GENERATE ANSWER USING OPENROUTER
# ============================================================

def answer_with_openrouter(question, evidence):

    import requests

    api_key = os.getenv(

        "OPENROUTER_API_KEY",

        ""

    )

    if not api_key:

        return (
            "Please configure OPENROUTER_API_KEY "
            "in Streamlit Secrets."
        )

    # Only use chunks classified as relevant

    relevant_evidence = [

        item for item in evidence

        if item.get("is_relevant", True)

    ]

    if not relevant_evidence:

        return (
            "The retrieved evidence does not provide "
            "enough relevant information to answer this question."
        )

    model = os.getenv(

        "OPENROUTER_MODEL",

        "openai/gpt-4o-mini"

    )

    context = ""

    for item in relevant_evidence:

        context += (

            f"\nSECTION: {item['title']}\n"

            f"PAGE: {item.get('page')}\n"

            f"RELEVANCE SCORE: "
            f"{item.get('relevance_score', 'N/A')}\n"

            f"EVIDENCE:\n"

            f"{item['text'][:4000]}\n"

        )

    system_prompt = """

You are a financial annual report question-answering assistant.

Rules:

1. Answer only using the supplied evidence.

2. Never invent financial figures.

3. Carefully distinguish turnover, revenue, CSR expenditure,
   environmental expenditure and order book.

4. Preserve the exact units, reporting year and company scope.

5. If the evidence does not contain the answer, say:
   "The retrieved evidence does not provide enough information."

6. Mention the PDF page number when available.

7. Do not use outside knowledge.

8. If multiple chunks contain related figures, carefully
   distinguish them instead of mixing their values.

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

                    f"RETRIEVED EVIDENCE:\n{context}"

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

    return response.json()["choices"][0]["message"]["content"]
