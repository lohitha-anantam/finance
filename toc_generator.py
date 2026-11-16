
"""
Finance STAIR - Table of Contents Generator

Functions:
1. Detect section headings.
2. Build a Table of Contents.
3. Divide the report into sections.
4. Preserve available PDF page markers.
"""

import re

PAGE_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)


def clean_heading(line):
    return re.sub(r"\s+", " ", line).strip(" \t:-–—|")


def is_heading(line):
    line = clean_heading(line)

    if not line:
        return False

    if len(line) < 3 or len(line) > 120:
        return False

    words = line.split()

    if len(words) > 14:
        return False

    # Avoid normal sentences.
    if re.search(r"[.!?;]$", line):
        return False

    # Numbered headings such as 1, 1.1, 2.3.4
    numbered = re.match(
        r"^(?:\d+(?:\.\d+){0,4}|[A-Z])[\).]?\s+\S",
        line
    )

    letters = re.sub(r"[^A-Za-z]", "", line)

    if not letters:
        return False

    uppercase_ratio = (
        sum(char.isupper() for char in letters) / len(letters)
    )

    title_case_count = sum(
        word[0].isupper()
        for word in words
        if word[0].isalpha()
    )

    title_ratio = title_case_count / max(1, len(words))

    return bool(
        numbered
        or uppercase_ratio > 0.78
        or (
            title_ratio > 0.75
            and len(words) <= 10
        )
    )


def build_toc(text, max_headings=250):

    headings = []
    seen = set()

    for line in text.splitlines():

        heading = clean_heading(line)

        if is_heading(heading):

            key = heading.lower()

            if key not in seen:

                seen.add(key)

                headings.append({
                    "title": heading,
                    "level": 1
                })

            if len(headings) >= max_headings:
                break

    return headings


def get_heading_title(item):

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        return str(item.get("title", "")).strip()

    return ""


def attach_sections(text, toc):
    """
    Divide the report into sections using detected headings.

    Returns:
        title
        text
        page_start
        page_end
        start_line
        end_line
    """

    lines = text.splitlines()

    heading_lookup = {}

    for item in toc:

        title = get_heading_title(item)

        if title:

            key = re.sub(
                r"\s+",
                " ",
                title
            ).strip().lower()

            heading_lookup[key] = title

    starts = []
    current_page = None

    for index, line in enumerate(lines):

        page_match = PAGE_RE.search(line)

        if page_match:
            current_page = int(page_match.group(1))

        cleaned = clean_heading(line)

        key = re.sub(
            r"\s+",
            " ",
            cleaned
        ).strip().lower()

        if key in heading_lookup:

            starts.append({
                "line": index,
                "title": heading_lookup[key],
                "page": current_page
            })

    sections = []

    for i, item in enumerate(starts):

        start = item["line"]

        if i + 1 < len(starts):
            end = starts[i + 1]["line"]
        else:
            end = len(lines)

        section_text = "\n".join(
            lines[start + 1:end]
        ).strip()

        if len(section_text) < 80:
            continue

        sections.append({
            "title": item["title"],
            "text": section_text,
            "page_start": item["page"],
            "page_end": None,
            "start_line": start + 1,
            "end_line": end
        })

    return sections


def toc_text(toc):

    output = []

    for item in toc:

        title = get_heading_title(item)

        if title:
            output.append(f"- {title}")

    return "\n".join(output)
