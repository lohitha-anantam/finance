from pathlib import Path
import fitz
from docx import Document


def read_file(file_path):
    """
    Extract document text while preserving PDF page numbers.
    Supports PDF, TXT, MD and DOCX.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        pdf = fitz.open(file_path)
        pages = []

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            pages.append(
                f"\n[PAGE {page_number}]\n{text}"
            )

        pdf.close()
        return "\n".join(pages)

    elif extension in [".txt", ".md"]:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    elif extension == ".docx":
        document = Document(file_path)
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )
