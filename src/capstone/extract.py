import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED = {".txt", ".pdf", ".docx"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        return "\n\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise ValueError(f"Unsupported file type: {ext or filename}")
