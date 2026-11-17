import os
import json
import time
import argparse
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """
You are a financial document question-generation assistant.

Generate questions that can be answered using ONLY the supplied section.

Rules:
1. Generate clear and realistic financial questions.
2. Include numerical, factual and analytical questions.
3. Do not invent information.
4. Do not use information outside the supplied section.
5. Questions must identify the subject clearly.
6. Return valid JSON only.

Required format:
{
  "questions": [
    "What was the company's revenue?",
    "What was its operating profit?"
  ]
}
"""


def create_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY before running."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def generate_questions(
    client,
    section,
    model,
    count=5
):
    prompt = f"""
Generate {count} questions using the following
financial report section.

Section title:
{section['title']}

Section path:
{section['path']}

Section content:
{section['text'][:12000]}

Return only JSON.
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    # Remove optional Markdown JSON fences.
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    result = json.loads(content)

    questions = result.get("questions", [])

    return [
        question.strip()
        for question in questions
        if isinstance(question, str)
        and question.strip()
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/corpus.jsonl"
    )

    parser.add_argument(
        "--output",
        default="data/processed/stair_train.jsonl"
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini"
    )

    parser.add_argument(
        "--questions-per-section",
        type=int,
        default=5
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {input_path}"
        )

    client = create_client()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_examples = 0

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as input_file, output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            if not line.strip():
                continue

            document = json.loads(line)

            doc_id = document["doc_id"]

            complete_toc = json.dumps(
                document["toc"],
                ensure_ascii=False
            )

            for section in document["sections"]:

                try:
                    questions = generate_questions(
                        client=client,
                        section=section,
                        model=args.model,
                        count=args.questions_per_section
                    )

                    for question in questions:

                        record = {
                            "doc_id": doc_id,
                            "question": question,
                            "toc": complete_toc,
                            "gold_leaf": section["path"],
                            "section_text": section["text"],
                            "page_start": section["page_start"],
                            "page_end": section["page_end"]
                        }

                        output_file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        total_examples += 1

                    print(
                        f"Generated {len(questions)} questions "
                        f"for {section['path']}"
                    )

                    time.sleep(0.5)

                except Exception as error:
                    print(
                        f"Skipping {section['path']}: {error}"
                    )

    print(
        f"\nTotal training examples: {total_examples}"
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
