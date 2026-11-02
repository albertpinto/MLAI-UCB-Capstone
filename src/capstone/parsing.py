"""Stage 1 and 2 of chunking: parent chunks (document structure) and child chunks (meaning boundaries)."""
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .chunking import chunk_text, clean_text, is_contents
from .config import Settings
from .extract import extract_text

logger = logging.getLogger(__name__)

SKIP_LABELS = {"footnote", "page_header", "page_footer"}  # page furniture: kept out of the text so it cannot pollute chunks
MIN_PARENT_CHARS = 400  # a heading starts a new parent only once the current one is at least this big
RUNNING_MIN_PAGES = 4  # a heading repeated on this many pages is a page header, not a section heading
RUNNING_MIN_SPAN = 0.5  # ...unless it recurs across half the document's page range: that is the document title, which says nothing about a part
# Footnote lines Docling leaves in the body: "Ins. by ...", "by the Constitution (... Amendment) Act ...", "4 (w.e.f. 13-4-1978)."
FOOTNOTE_LINE = re.compile(r"^(\d+\s*)?(Ins|Subs|Added|Re-numbered)\.(\s|$)|^(\d+\s*)?by the Constitution \(|w\.e\.f\.?[\s\d\-.)]*$", re.I)
# A bare article number ("21." or "2 [21A.") that ends a piece belongs to the sentence after it: the splitter cut at its full stop.
NUMBER_LINE = re.compile(r"^(\d+\s*\[\s*)?\d+[A-Z]{0,2}\.$")


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


def _label(item) -> str:
    return getattr(item.label, "value", str(item.label))


def _page(item) -> int | None:
    return item.prov[0].page_no if getattr(item, "prov", None) else None


def _header_key(text: str) -> str:
    """A running header with its page number removed: "THE CONSTITUTION OF INDIA 144" -> "THE CONSTITUTION OF INDIA"."""
    return re.sub(r"^\d+\s+|\s+\d+$", "", text.strip())


def strip_footnotes(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not FOOTNOTE_LINE.search(line.strip()))


PART = re.compile(r"^\(?\s*part\s+([ivxlc]+[a-z]?)\b[\s.\-–—:]*(.*?)\)?$", re.I)


def _part(text: str) -> tuple[str, str] | None:
    """"(Part V.-The Union)" and "PART V THE UNION" are both Part V: (key, label), whatever the case or punctuation."""
    m = PART.match(text.strip())
    if not m:
        return None
    name = m.group(2).strip(" .-–—")
    name = name.title().replace(" Of ", " of ").replace(" And ", " and ").replace(" The ", " the ") if name.isupper() else name
    return m.group(1).upper(), f"Part {m.group(1).upper()}" + (f" – {name}" if name else "")


def _heading(text: str) -> str:
    """"(Part V.-The Union)" -> "Part V – The Union"."""
    text = re.sub(r"^\((.*)\)$", r"\1", text.strip())
    return re.sub(r"\s+", " ", text.replace(".-", " – ")).strip()


def page_headers(items: list) -> tuple[set[str], set[str]]:
    """Headings that repeat on many pages are page furniture. Returns (running headers that name a part, the document title)."""
    pages: dict[str, set[int]] = defaultdict(set)
    all_pages: set[int] = set()
    for item, _ in items:
        page = _page(item)
        if page is not None:
            all_pages.add(page)
            if _label(item) == "section_header":
                pages[_header_key(item.text or "")].add(page)
    total = max(max(all_pages, default=1) - min(all_pages, default=1), 1)
    repeated = {t for t, p in pages.items() if t and len(p) >= RUNNING_MIN_PAGES}
    titles = {t for t in repeated if max(pages[t]) - min(pages[t]) > RUNNING_MIN_SPAN * total}
    return repeated - titles, titles


def _docling_parents(name: str, data: bytes, settings: Settings) -> list[Parent]:
    from docling.datamodel.base_models import DocumentStream
    from io import BytesIO

    document = _converter(settings.docling_threads).convert(DocumentStream(name=name, stream=BytesIO(data))).document
    items = list(document.iterate_items())
    running, titles = page_headers(items)
    parents: list[Parent] = []
    headings: list[str] = []
    parts: list[str] = []
    part, part_key, section, page = "", "", "", None

    def flush():
        nonlocal parts, section, page
        text = "\n".join(parts).strip()
        if text and not garbled(text) and not is_contents(text):  # a table of contents answers nothing and would be rewritten into prose
            parents.append(Parent(text, section, page))
        parts, section, page = [], "", None

    for item, level in items:
        label = _label(item)
        text = (getattr(item, "text", "") or "").strip()
        if label != "section_header":
            text = strip_footnotes(text).strip()
        if label in SKIP_LABELS or not text or re.fullmatch(r"[_\-=\s]{5,}", text) or _header_key(text) in titles:
            continue
        if label == "section_header":
            info = _part(text) or ((_heading(_header_key(text)),) * 2 if _header_key(text) in running else None)  # a Part heading starts a new Part, however rarely it repeats
            if info:
                if info[0] != part_key:
                    flush()
                    part_key, part, headings = info[0], info[1], []
            else:
                if sum(map(len, parts)) >= MIN_PARENT_CHARS:
                    flush()
                headings = headings[: max(level - 1, 0)] + [_heading(text)]
            continue
        if parts and sum(map(len, parts)) + len(text) > settings.parent_chars:
            flush()
        if not parts:
            section = " › ".join(h for h in [part, *headings] if h)
            page = _page(item)
        parts.append(text)
    flush()
    return parents


@lru_cache
def _chunker(words: int, embed_model: str):
    from chonkie import SemanticChunker

    return SemanticChunker(embedding_model=embed_model, chunk_size=words, tokenizer="word", min_characters_per_sentence=10)


def _cap(piece: str, limit: int, target: int) -> list[str]:
    """Chonkie cannot split inside a sentence, so a very long one (a list, a clause chain) is cut at line ends and
    semicolons into pieces of about `target` words."""
    if len(piece.split()) <= limit:
        return [piece]
    out, current = [], ""
    for part in re.split(r"(?<=[;:.])\s+|\n", piece):
        if current and len((current + " " + part).split()) > target:
            out.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    return out + ([current] if current else [])


def move_numbers(pieces: list[str]) -> list[str]:
    """Carry a bare article number left at the end of a piece to the start of the next one."""
    out, carry = [], ""
    for i, piece in enumerate(pieces):
        piece = f"{carry}\n{piece}".strip() if carry else piece
        lines, carry = piece.split("\n"), ""
        moved = []
        while lines and NUMBER_LINE.match(lines[-1].strip()):
            moved.insert(0, lines.pop().strip())
        if moved and i < len(pieces) - 1:
            carry = "\n".join(moved)
        else:
            lines += moved
        if lines:
            out.append("\n".join(lines))
    if carry:
        out.append(carry)
    return out


def split_children(parent_text: str, settings: Settings) -> list[str]:
    """Split a parent into child chunks at semantic boundaries, then merge pieces that are too small to embed well."""
    pieces = [c.text.strip() for c in _chunker(settings.child_words, settings.embed_model).chunk(parent_text) if c.text.strip()]
    pieces = move_numbers(pieces)
    pieces = [small for piece in pieces for small in _cap(piece, settings.child_words * 2, settings.child_words)]
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
