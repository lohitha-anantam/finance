
from pathlib import Path
import fitz
from docx import Document


# ============================================================
# 1. EXTRACT TEXT FROM PDF
# ============================================================

def read_pdf(file_path):
    """
    Extract text from every PDF page.

    Each page is marked as:
    [PAGE 1]
    Page content...

    These markers are required by stair_core.py
    to preserve PDF page references.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file is not a PDF.")

    pages = []

    try:
        with fitz.open(str(path)) as pdf:

            if pdf.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            for page_number, page in enumerate(pdf, start=1):

                page_text = page.get_text("text") or ""

                # Clean extracted text without removing content
                page_text = page_text.strip()

                pages.append(
                    f"\n[PAGE {page_number}]\n"
                    f"{page_text}\n"
                )

    except Exception as error:
        raise RuntimeError(
            f"Unable to extract PDF text: {error}"
        ) from error

    extracted_text = "\n".join(pages)

    if not extracted_text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned PDF requiring OCR."
        )

    return extracted_text


# ============================================================
# 2. EXTRACT TEXT FROM TXT AND MARKDOWN
# ============================================================

def read_text_file(file_path):
    """
    Read TXT or Markdown files.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# ============================================================
# 3. EXTRACT TEXT FROM DOCX
# ============================================================

def read_docx(file_path):
    """
    Extract paragraphs and tables from DOCX files.

    DOCX documents do not have reliable PDF page numbers,
    so page references are not generated here.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"DOCX file not found: {file_path}"
        )

    document = Document(str(path))

    content = []

    # Extract paragraphs
    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract tables
    for table in document.tables:

        for row in table.rows:

            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
            )

            if row_text.strip():
                content.append(row_text)

    return "\n".join(content)


# ============================================================
# 4. MAIN DOCUMENT READER
# ============================================================

def read_file(file_path):
    """
    Main document extraction function.

    Supported formats:
    PDF, TXT, MD and DOCX.

    PDF page numbers are preserved using [PAGE n] markers.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":

        return read_pdf(path)

    elif extension in [".txt", ".md"]:

        return read_text_file(path)

    elif extension == ".docx":

        return read_docx(path)

    else:

        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats: PDF, TXT, MD, DOCX."
        )


# ============================================================
# 5. CHECK PDF PAGE MARKERS
# ============================================================

def get_page_marker_count(text):
    """
    Count extracted PDF page markers.
    """

    import re

    markers = re.findall(
        r"\[PAGE\s+\d+\]",
        text
    )

    return len(markers)


# ============================================================
# 6. GET PAGE NUMBERS FROM EXTRACTED TEXT
# ============================================================

def get_extracted_page_numbers(text):
    """
    Return all page numbers present in extracted text.
    """

    import re

    markers = re.findall(
        r"\[PAGE\s+(\d+)\]",
        text
    )

    return [int(number) for number in markers]


# ============================================================
# 7. DOCUMENT EXTRACTION DIAGNOSTICS
# ============================================================

def get_document_diagnostics(text):
    """
    Return basic extraction information.
    """

    page_numbers = get_extracted_page_numbers(text)

    return {

        "total_characters": len(text),

        "page_marker_count": len(page_numbers),

        "page_numbers": page_numbers,

        "has_page_markers": len(page_numbers) > 0,

        "extraction_successful": bool(text.strip())

    }
