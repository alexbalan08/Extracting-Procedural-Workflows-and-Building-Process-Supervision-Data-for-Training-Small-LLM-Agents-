"""PDF -> plain text for the demo app.

Step 1 of the pipeline: take an uploaded PDF and recover the procedure text.
We try PyMuPDF (fitz) first — it is fast and keeps reading order well — and
fall back to pdfplumber if fitz is unavailable or returns nothing useful.
Neither needs a network call or a GPU, so this runs live on the Mac.
"""

from __future__ import annotations

import io
import re


def _clean(text: str) -> str:
    # collapse the ragged whitespace PDFs leave behind without destroying
    # paragraph boundaries — extractors downstream read this as the procedure body
    text = text.replace("\r", "\n")
    # join hyphenated line breaks: "extrac-\ntion" -> "extraction"
    text = re.sub(r"-\n(?=\w)", "", text)
    # single newlines inside a paragraph -> space; keep blank-line paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_pymupdf(data: bytes) -> tuple[str, list[str]]:
    import fitz  # PyMuPDF

    pages: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return _clean("\n\n".join(pages)), pages


def extract_text_pdfplumber(data: bytes) -> tuple[str, list[str]]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return _clean("\n\n".join(pages)), pages


def extract_text(data: bytes) -> tuple[str, dict]:
    """Return (cleaned_text, meta). meta records which parser won and page count."""
    errors: dict[str, str] = {}
    for name, fn in (("pymupdf", extract_text_pymupdf), ("pdfplumber", extract_text_pdfplumber)):
        try:
            text, pages = fn(data)
            if text.strip():
                return text, {"parser": name, "n_pages": len(pages), "errors": errors}
        except Exception as e:  # noqa: BLE001 - surface to the UI, don't crash
            errors[name] = f"{type(e).__name__}: {e}"
    return "", {"parser": None, "n_pages": 0, "errors": errors}
