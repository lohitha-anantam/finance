import os
import json
import argparse
from pathlib import Path

from document_parser import read_file
from toc_generator import (
    build_toc,
    flatten_nodes,
    leaf_nodes,
    attach_sections
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx"
}


def process_document(file_path):
    print(f"\nProcessing: {file_path.name}")

    text = read_file(file_path)

    if not text.strip():
        print("Skipping empty document.")
        return None

    toc = build_toc(text)

    if not toc:
        print("No headings detected. Review document formatting.")
        return None

    sections = attach_sections(text, toc)

    if not sections:
        print("No sections extracted.")
        return None

    valid_leaf_paths = {
        node["path"]
        for node in leaf_nodes(toc)
    }

    # Keep only sections mapped to valid leaf nodes.
    sections = [
        section
        for section in sections
        if section["path"] in valid_leaf_paths
    ]

    return {
        "doc_id": file_path.stem,
        "filename": file_path.name,
        "toc": toc,
        "toc_text": "\n".join(
            node["path"]
            for node in leaf_nodes(toc)
        ),
        "sections": sections
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/reports"
    )

    parser.add_argument(
        "--output",
        default="data/processed/corpus.jsonl"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    files = [
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            "No supported reports found in input folder."
        )

    processed_count = 0

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for file_path in files:

            try:
                document = process_document(file_path)

                if document is None:
                    continue

                output_file.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    ) + "\n"
                )

                processed_count += 1

                print(
                    f"Extracted {len(document['sections'])} sections."
                )

            except Exception as error:
                print(
                    f"Error processing {file_path.name}: {error}"
                )

    print(
        f"\nCompleted. Processed {processed_count} documents."
    )

    print(
        f"Output saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
