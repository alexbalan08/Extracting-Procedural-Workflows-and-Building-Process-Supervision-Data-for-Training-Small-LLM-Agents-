
from __future__ import annotations

import io
import re


def _clean(text: str) -> str:
    
    text = text.replace("\r", "\n")
    
    text = re.sub(r"-\n(?=\w)", "", text)
    
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_pymupdf(data: bytes) -> tuple[str, list[str]]:
    import fitz  

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
    
    errors: dict[str, str] = {}
    for name, fn in (("pymupdf", extract_text_pymupdf), ("pdfplumber", extract_text_pdfplumber)):
        try:
            text, pages = fn(data)
            if text.strip():
                return text, {"parser": name, "n_pages": len(pages), "errors": errors}
        except Exception as e:  # noqa: BLE001 - surface to the UI, don't crash
            errors[name] = f"{type(e).__name__}: {e}"
    return "", {"parser": None, "n_pages": 0, "errors": errors}
