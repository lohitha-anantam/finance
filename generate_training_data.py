
"""
Finance STAIR - Training Data Generator

Input:
    data/structured.jsonl

Output:
    data/stair_train.jsonl

Generates:
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

    for item in toc or []:
        visit(item)

    # Remove duplicate headings

    result = []
    seen = set()

    for title in output:

        key = title.lower()

        if key not in seen:

            seen.add(key)
            result.append(title)

    return result


# --------------------------------------------------
# 2. OpenRouter System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You generate high-quality training questions for a
financial annual-report retrieval system.

Use ONLY the supplied section text.

Generate exactly TWO different questions.

Questions should cover useful financial information such as:

- Revenue and turnover
- Profit and loss
- EBITDA
- Total assets
- Debt and borrowings
- Cash flow
- Dividends
- Employee information
- Business performance
- Risks and strategies
- Financial year information

Rules:

1. Questions must be natural and realistic.
2. Questions must be answerable from the supplied section.
3. Answers must be supported by the supplied text.
4. Preserve exact financial figures and units.
5. Include the financial year when available.
6. Never invent numbers or facts.
7. Do not ask questions that require other sections.

Return ONLY a valid JSON array.

Format:

[
  {
    "question": "What was the reported revenue?",
    "answer": "The reported revenue was ..."
  }
]

Do not include Markdown.
Do not include explanations.
"""


# --------------------------------------------------
# 3. Parse JSON Response
# --------------------------------------------------

def extract_json_array(content):

    content = content.strip()

    # Remove Markdown code fences

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

        raise ValueError(
            "OpenRouter did not return a JSON array."
        )

    return json.loads(
        content[start:end + 1]
    )


# --------------------------------------------------
# 4. Generate Questions Using OpenRouter
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Validate Generated Questions
# --------------------------------------------------

def validate_examples(items, section_title):

    valid = []

    for item in items:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        answer = str(
            item.get("answer", "")
        ).strip()

        if len(question) < 10:
            continue

        if len(answer) < 3:
            continue

        valid.append({

            "question": question,

            "answer": answer,

            "gold_leaf": section_title

        })

    return valid


# --------------------------------------------------
# 6. Normalize Questions
# --------------------------------------------------

def normalize_question(question):

    return re.sub(
        r"\s+",
        " ",
        question.strip().lower()
    )


# --------------------------------------------------
# 7. Generate Training Dataset
# --------------------------------------------------

def main(

    source="data/structured.jsonl",

    output="data/stair_train.jsonl",

    limit=500

):

    source_path = Path(source)

    output_path = Path(output)

    if not source_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {source}"
        )

    if limit <= 0:

        raise ValueError(
            "Training data limit must be greater than zero."
        )

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

        for line_number, line in enumerate(
            input_file,
            1
        ):

            if count >= limit:
                break

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                print(
                    f"Skipping invalid JSON line {line_number}"
                )

                continue

            doc_id = record.get(
                "doc_id",
                f"document_{line_number}"
            )

            sections = record.get(
                "sections",
                []
            )

            toc = record.get(
                "toc",
                []
            )

            leaf_titles = flatten_toc(toc)

            if not leaf_titles:

                print(
                    f"Skipping {doc_id}: ToC is empty."
                )

                continue

            # Use the complete list of ToC leaf titles.

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

                if not title:
                    continue

                if len(section_text) < 100:
                    continue

                # Validate section title against ToC.

                matched_title = next(

                    (

                        leaf

                        for leaf in leaf_titles

                        if leaf.lower() == title.lower()

                    ),

                    None

                )

                if matched_title is None:

                    print(
                        f"Skipping section not found in ToC: {title}"
                    )

                    continue

                # Generate synthetic questions.

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
                        f"Question generation failed "
                        f"for {matched_title}: {error}"
                    )

                    continue

                added = 0

                for example in examples:

                    if count >= limit:
                        break

                    question = example["question"]

                    key = normalize_question(
                        question
                    )

                    # Avoid duplicate questions.

                    if key in seen_questions:
                        continue

                    seen_questions.add(key)

                    output_record = {

                        "question": question,

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
                    added += 1

                    if added >= 2:
                        break

                print(
                    f"{matched_title}: "
                    f"{added} examples generated"
                )

    print("\nTraining data generation completed.")

    print(
        f"Total training examples: {count}"
    )

    print(
        f"Output file: {output_path}"
    )


# --------------------------------------------------
# 8. Run Script
# --------------------------------------------------

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
        default=500
    )

    args = parser.parse_args()

    main(

        source=args.source,

        output=args.output,

        limit=args.limit

    )
