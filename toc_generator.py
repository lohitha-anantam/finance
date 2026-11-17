import re
from collections import Counter

PAGE_RE = re.compile(
    r"\[PAGE\s+(\d+)\]",
    re.I
)

NUM_RE = re.compile(
    r"^(\d+(?:\.\d+){0,5})[.)]?\s+(.+?)\s*$"
)

APPENDIX_RE = re.compile(
    r"^(Appendix\s+[A-Z0-9]+[:.\-]?\s*.+)$",
    re.I
)


def clean_line(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip(" \t|")


def heading_level(line):
    """
    Estimate heading level using numbering,
    capitalization and title formatting.
    """
    text = clean_line(line)

    if (
        not text
        or len(text) > 150
        or len(text.split()) > 18
        or re.search(r"[.!?]$", text)
    ):
        return None

    match = NUM_RE.match(text)

    if match:
        return min(
            match.group(1).count(".") + 1,
            6
        )

    if APPENDIX_RE.match(text):
        return 1

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return None

    upper_ratio = (
        sum(char.isupper() for char in letters)
        / len(letters)
    )

    if upper_ratio >= 0.82 and 2 <= len(text.split()) <= 14:
        return 1

    words = text.split()

    title_ratio = (
        sum(
            word[0].isupper()
            for word in words
            if word[0].isalpha()
        )
        / max(1, len(words))
    )

    if (
        title_ratio >= 0.8
        and len(words) <= 10
        and len(text) <= 90
    ):
        return 2

    return None


def build_toc(text, max_headings=500):
    """
    Build a nested Table of Contents.
    Review the generated hierarchy for each report.
    """
    roots = []
    stack = []
    seen = set()

    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        level = heading_level(line)

        if not level:
            continue

        key = line.casefold()

        if key in seen:
            continue

        seen.add(key)

        node = {
            "title": line,
            "level": level,
            "children": []
        }

        while (
            stack
            and stack[-1]["level"] >= level
        ):
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)

        stack.append(node)

        if len(seen) >= max_headings:
            break

    return roots


def flatten_nodes(toc):
    result = []

    def walk(nodes, parent=""):
        for node in nodes or []:

            if isinstance(node, str):
                path = (
                    f"{parent} > {node}"
                    if parent else node
                )

                result.append({
                    "title": node,
                    "path": path,
                    "level": 1
                })

                continue

            title = clean_line(
                str(node.get("title", ""))
            )

            path = (
                f"{parent} > {title}"
                if parent else title
            )

            children = node.get("children", [])

            result.append({
                "title": title,
                "path": path,
                "level": node.get("level", 1),
                "is_leaf": not bool(children)
            })

            walk(children, path)

    walk(toc)

    return result


def leaf_nodes(toc):
    return [
        node
        for node in flatten_nodes(toc)
        if node.get("is_leaf") and node["title"]
    ]


def toc_text(toc):
    lines = []

    for node in flatten_nodes(toc):
        indentation = (
            "  " * (
                int(node.get("level", 1)) - 1
            )
        )

        lines.append(
            indentation + node["title"]
        )

    return "\n".join(lines)


def attach_sections(text, toc):
    """
    Match headings with their corresponding content.
    Preserve PDF page references.
    """
    flat = flatten_nodes(toc)

    by_title = {}

    for node in flat:
        key = clean_line(
            node["title"]
        ).casefold()

        by_title.setdefault(
            key, []
        ).append(node)

    lines = text.splitlines()

    heading_hits = []
    current_page = None

    for index, raw_line in enumerate(lines):

        page_match = PAGE_RE.search(raw_line)

        if page_match:
            current_page = int(
                page_match.group(1)
            )

        key = clean_line(
            raw_line
        ).casefold()

        if key in by_title:
            heading_hits.append({
                "line": index,
                "nodes": by_title[key],
                "page": current_page
            })

    sections = []
    used = Counter()

    for index, hit in enumerate(heading_hits):

        options = hit["nodes"]

        key = clean_line(
            lines[hit["line"]]
        ).casefold()

        selected = options[
            min(
                used[key],
                len(options) - 1
            )
        ]

        used[key] += 1

        end_line = (
            heading_hits[index + 1]["line"]
            if index + 1 < len(heading_hits)
            else len(lines)
        )

        body = "\n".join(
            lines[hit["line"] + 1:end_line]
        ).strip()

        if len(body) < 40:
            continue

        end_page = hit["page"]

        for line in lines[
            hit["line"] + 1:end_line
        ]:
            page_match = PAGE_RE.search(line)

            if page_match:
                end_page = int(
                    page_match.group(1)
                )

        sections.append({
            "title": selected["title"],
            "path": selected["path"],
            "text": body,
            "page_start": hit["page"],
            "page_end": end_page,
            "start_line": hit["line"] + 1,
            "end_line": end_line
        })

    return sections
