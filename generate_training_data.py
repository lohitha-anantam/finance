
"""
Generate synthetic STAIR training data using OpenRouter.

Input:
    data/structured.jsonl

Output:
    data/stair_train.jsonl

Each output record contains:
    question
    answer
    gold_leaf
    doc_id
    toc
"""

import os
import re
import json
import argparse
import requests

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL


# ---------------------------------------------------------
# 1. ToC utilities
# ---------------------------------------------------------

def flatten_toc(toc):

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

    result = []
    seen = set()

    for title in leaves:
        key = title.lower()

        if title and key not in seen:
            seen.add(key)
            result.append(title)

    return result


# ---------------------------------------------------------
# 2. OpenRouter prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You create high-quality training questions for a financial
annual-report retrieval system.

Use ONLY the supplied section text.

Generate exactly TWO different questions.

Each question must:
- Be natural and realistic.
- Be answerable from the supplied text.
- Focus on a meaningful fact, financial metric, year, risk,
  strategy, performance indicator or business detail.
- Avoid questions that require information outside the section.

Each answer must:
- Be short and factually supported by the supplied text.
- Preserve the exact financial metric and units.
- Include the financial year when available.
- Never invent numbers or facts.

Return ONLY a valid JSON array.

Each object must contain:
{
  "question": "question text",
  "answer": "evidence-based answer"
}

Do not include gold_leaf.
Do not include Markdown.
"""


def extract_json_array(content):

    content = content.strip()

    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    start = content.find("[")
    end = content.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model response.")

    return json.loads(content[start:end + 1])


def ask_openrouter(section_text, section_title):

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in your .env file."
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"SECTION TITLE: {section_title}\n\n"
                    f"SECTION TEXT:\n{section_text[:10000]}"
                )
            }
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    return extract_json_array(content)


# ---------------------------------------------------------
# 3. Validate generated examples
# ---------------------------------------------------------

def normalize_question(question):

    return re.sub(
        r"\s+",
        " ",
        str(question).strip().lower()
    )


def validate_examples(items, section_title):

    valid = []

    for item in items:

        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if len(question) < 10 or len(answer) < 3:
            continue

        valid.append({
            "question": question,
            "answer": answer,
            "gold_leaf": section_title
        })

    return valid


# ---------------------------------------------------------
# 4. Generate training dataset
# ---------------------------------------------------------

def main(
    source="data/structured.jsonl",
    output="data/stair_train.jsonl",
    limit=100,
    questions_per_section=2
):

    source_path = Path(source)
    output_path = Path(output)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {source}"
        )

    if limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0
    seen_questions = set()

    with open(
        source_path,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line_number, line in enumerate(input_file, 1):

            if count >= limit:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid JSON at line {line_number}"
                )
                continue

            doc_id = record.get(
                "doc_id",
                f"document_{line_number}"
            )

            sections = record.get("sections", [])
            toc = record.get("toc", [])

            leaf_titles = flatten_toc(toc)

            if not leaf_titles:
                print(
                    f"Skipping {doc_id}: no ToC leaf titles found."
                )
                continue

            # Use the complete available leaf-title list.
            training_toc = leaf_titles

            for section in sections:

                if count >= limit:
                    break

                title = str(
                    section.get("title", "")
                ).strip()

                section_text = str(
                    section.get("text", "")
                ).strip()

                if not title or len(section_text) < 100:
                    continue

                # Only train on actual ToC leaf sections.
                matched_title = next(
                    (
                        leaf for leaf in leaf_titles
                        if leaf.lower() == title.lower()
                    ),
                    None
                )

                if matched_title is None:
                    print(
                        f"Skipping section not found in ToC: {title}"
                    )
                    continue

                try:
                    generated = ask_openrouter(
                        section_text,
                        matched_title
                    )

                    examples = validate_examples(
                        generated,
                        matched_title
                    )

                except Exception as error:
                    print(
                        f"Generation failed for {title}: {error}"
                    )
                    continue

                added_for_section = 0

                for example in examples:

                    if count >= limit:
                        break

                    if added_for_section >= questions_per_section:
                        break

                    question_key = normalize_question(
                        example["question"]
                    )

                    if question_key in seen_questions:
                        continue

                    seen_questions.add(question_key)

                    output_record = {
                        "question": example["question"],
                        "answer": example["answer"],
                        "gold_leaf": matched_title,
                        "doc_id": doc_id,
                        "toc": training_toc
                    }

                    output_file.write(
                        json.dumps(
                            output_record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    output_file.flush()

                    count += 1
                    added_for_section += 1

                print(
                    f"{title}: generated {added_for_section} examples"
                )

    print("\nTraining data generation completed.")
    print(f"Total examples: {count}")
    print(f"Output file: {output_path}")


# ---------------------------------------------------------
# 5. Command line arguments
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="data/structured.jsonl"
    )

    parser.add_argument(
        "--output",
        default="data/stair_train.jsonl"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100
    )

    parser.add_argument(
        "--questions-per-section",
        type=int,
        default=2
    )

    args = parser.parse_args()

    main(
        source=args.source,
        output=args.output,
        limit=args.limit,
        questions_per_section=args.questions_per_section
    )
