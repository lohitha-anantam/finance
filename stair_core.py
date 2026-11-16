
import os
import re
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from toc_generator import attach_sections, toc_text

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 1. Load semantic embedding model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


# --------------------------------------------------
# 2. Flatten hierarchical ToC
# --------------------------------------------------

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

                for key in ("children", "subsections", "sections"):
                    if isinstance(node.get(key), list):
                        visit(node[key])

    visit(toc)
    return list(dict.fromkeys(result))


# --------------------------------------------------
# 3. Predict relevant ToC section
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Create page-based chunks
# --------------------------------------------------

def create_page_chunks(text, chunk_size=1200, overlap=200):

    pages = re.split(r"\[PAGE\s+(\d+)\]", text)
    chunks = []

    # Text containing explicit page markers
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

    # Fallback if parser did not preserve page markers
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


# --------------------------------------------------
# 5. Keyword matching
# --------------------------------------------------

def keyword_score(query, document):

    query_words = set(re.findall(r"[a-zA-Z0-9₹]+", query.lower()))
    doc_words = set(re.findall(r"[a-zA-Z0-9₹]+", document.lower()))

    if not query_words:
        return 0.0

    return len(query_words & doc_words) / len(query_words)


# --------------------------------------------------
# 6. Financial metric matching
# --------------------------------------------------

def metric_bonus(question, document):

    q = question.lower()
    d = document.lower()

    metric_groups = {
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
        ]
    }

    bonus = 0.0

    for terms in metric_groups.values():
        if any(term in q for term in terms):
            if any(term in d for term in terms):
                bonus += 0.20

    return min(bonus, 0.40)


# --------------------------------------------------
# 7. Hybrid retrieval
# --------------------------------------------------

def retrieve_evidence(
    question,
    text,
    toc,
    predicted_leaf=None,
    top_k=5
):

    candidates = []

    # A. Extract ToC-based sections
    try:
        sections = attach_sections(text, toc)

        for section in sections:
            section_text = section.get("text", "")

            if section_text.strip():
                candidates.append({
                    "title": section.get("title", "Document Section"),
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
    unique = {}
    for item in candidates:
        key = item["text"][:500].strip().lower()
        if key and key not in unique:
            unique[key] = item

    candidates = list(unique.values())

    documents = [
        item["title"] + " " + item["text"]
        for item in candidates
    ]

    # D. Semantic similarity
    semantic_scores = np.zeros(len(documents))

    try:
        embedder = load_embedder()

        query_embedding = embedder.encode(
            [question],
            normalize_embeddings=True
        )

        doc_embeddings = embedder.encode(
            documents,
            normalize_embeddings=True
        )

        semantic_scores = (
            doc_embeddings @ query_embedding[0]
        )

        semantic_scores = np.clip(semantic_scores, 0, 1)

    except Exception as e:
        print("Semantic search unavailable:", e)

    # E. TF-IDF keyword similarity
    keyword_scores = np.zeros(len(documents))

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            [question] + documents
        )

        keyword_scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

    except Exception as e:
        print("Keyword search error:", e)

    # F. Combine scores
    results = []

    for i, item in enumerate(candidates):

        document = documents[i]

        exact_score = keyword_score(
            question,
            document
        )

        financial_score = metric_bonus(
            question,
            document
        )

        section_bonus = 0.0

        if predicted_leaf:
            if predicted_leaf.lower() in item["title"].lower():
                section_bonus = 0.15

        final_score = (
            0.40 * semantic_scores[i]
            + 0.35 * keyword_scores[i]
            + 0.15 * exact_score
            + financial_score
            + section_bonus
        )

        results.append({
            **item,
            "score": float(final_score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 8. Generate answer using OpenRouter
# --------------------------------------------------

def answer_with_openrouter(question, evidence):

    import requests

    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        return "Please configure OPENROUTER_API_KEY in Streamlit Secrets."

    if not evidence:
        return "No relevant evidence was retrieved. Please check the PDF extraction."

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    context = ""

    for item in evidence:
        context += (
            f"\nSECTION: {item['title']}\n"
            f"PAGE: {item.get('page')}\n"
            f"EVIDENCE:\n{item['text'][:4000]}\n"
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
