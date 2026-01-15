from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def extract_text_by_page(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return pages
