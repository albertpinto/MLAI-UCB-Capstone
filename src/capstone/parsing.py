"""Stage 1 and 2 of chunking: parent chunks (document structure) and child chunks (meaning boundaries)."""
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .chunking import chunk_text, clean_text, is_contents
from .config import Settings
from .extract import extract_text

logger = logging.getLogger(__name__)

SKIP_LABELS = {"footnote", "page_header", "page_footer"}  # page furniture: kept out of the text so it cannot pollute chunks
MIN_PARENT_CHARS = 400  # a heading starts a new parent only once the current one is at least this big


@dataclass
class Parent:
    text: str
    section: str = ""  # heading path, e.g. "Part III › Right to Freedom"
    page: int | None = None


def garbled(text: str) -> bool:
    """Legacy-font text (e.g. the Hindi pages of a diglot PDF) extracts as accented Latin gibberish."""
    return sum(1 for ch in text if 0x80 <= ord(ch) <= 0xFF) / max(len(text), 1) > 0.03


def parents_from_text(text: str, parent_chars: int) -> list[Parent]:
    """Plain text has no layout: pack whole paragraphs into parents of about `parent_chars`."""
    return [Parent(c) for c in chunk_text(text, parent_chars, 0)]


@lru_cache
def _converter(threads: int):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False,  # the PDFs have a text layer; OCR would load more models and run far slower
        do_table_structure=False,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=threads),
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def parents_from_file(name: str, data: bytes, settings: Settings) -> list[Parent]:
    """.txt: paragraphs. .pdf/.docx: Docling reads the layout, so headings group the text and footnotes are left out.
    If Docling fails, fall back to plain extracted text."""
    if Path(name).suffix.lower() == ".txt":
        return parents_from_text(clean_text(extract_text(name, data)), settings.parent_chars)
    try:
        return _docling_parents(name, data, settings)
    except Exception as exc:
        logger.error("Docling failed for %s (%s); using plain text extraction", name, exc)
        return parents_from_text(clean_text(extract_text(name, data)), settings.parent_chars)


def _docling_parents(name: str, data: bytes, settings: Settings) -> list[Parent]:
    from docling.datamodel.base_models import DocumentStream
    from io import BytesIO

    document = _converter(settings.docling_threads).convert(DocumentStream(name=name, stream=BytesIO(data))).document
    parents: list[Parent] = []
    headings: list[str] = []
    parts: list[str] = []
    section, page = "", None

    def flush():
        nonlocal parts, section, page
        text = "\n".join(parts).strip()
        if text and not garbled(text) and not is_contents(text):  # a table of contents answers nothing and would be rewritten into prose
            parents.append(Parent(text, section, page))
        parts, section, page = [], "", None

    for item, level in document.iterate_items():
        label = getattr(item.label, "value", str(item.label))
        text = (getattr(item, "text", "") or "").strip()
        if label in SKIP_LABELS or not text or re.fullmatch(r"[_\-=\s]{5,}", text):
            continue
        if label == "section_header":
            if sum(map(len, parts)) >= MIN_PARENT_CHARS:
                flush()
            headings = headings[: max(level - 1, 0)] + [text]
            continue
        if parts and sum(map(len, parts)) + len(text) > settings.parent_chars:
            flush()
        if not parts:
            section = " › ".join(headings)
            page = item.prov[0].page_no if getattr(item, "prov", None) else None
        parts.append(text)
    flush()
    return parents


@lru_cache
def _chunker(words: int, embed_model: str):
    from chonkie import SemanticChunker

    return SemanticChunker(embedding_model=embed_model, chunk_size=words, tokenizer="word", min_characters_per_sentence=10)


def split_children(parent_text: str, settings: Settings) -> list[str]:
    """Split a parent into child chunks at semantic boundaries, then merge pieces that are too small to embed well."""
    pieces = [c.text.strip() for c in _chunker(settings.child_words, settings.embed_model).chunk(parent_text) if c.text.strip()]
    children: list[str] = []
    for piece in pieces:
        if children and len(piece.split()) < settings.child_min_words:
            children[-1] = f"{children[-1]}\n{piece}"
        else:
            children.append(piece)
    if len(children) > 1 and len(children[0].split()) < settings.child_min_words:
        children[1] = f"{children[0]}\n{children[1]}"
        children = children[1:]
    return children
