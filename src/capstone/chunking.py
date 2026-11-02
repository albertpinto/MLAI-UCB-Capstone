import re


def clean_text(text: str) -> str:
    """Normalise line endings and collapse runs of spaces; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _split_long(paragraph: str, size: int) -> list[str]:
    """Cut a paragraph longer than `size`, preferring sentence ends."""
    pieces = []
    while len(paragraph) > size:
        cut = paragraph.rfind(". ", 0, size)
        cut = cut + 1 if cut > size // 2 else size
        pieces.append(paragraph[:cut].strip())
        paragraph = paragraph[cut:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of about `size` characters, carrying `overlap` characters forward."""
    pieces = []
    for paragraph in clean_text(text).split("\n\n"):
        pieces.extend(_split_long(paragraph.strip(), size))

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:].split(" ", 1)[-1] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


_NUMBERED = re.compile(r"^\s*\d+[A-Z]{0,2}\.\s")


def reflow(text: str) -> str:
    """Undo the hard line wraps PDFs leave in text, so it reads (and is sent to the LLM) as sentences.

    A line continues the previous one when it starts in lowercase, or when the previous line has no closing
    punctuation and the new line is not a list item such as "(a)" or "19.". Blank lines stay paragraph breaks.
    """
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if lines and lines[-1]:
            previous = lines[-1]
            is_item = line.startswith("(") or _NUMBERED.match(line)
            if line[0].islower() or (not previous.endswith((".", ";", ":", "?", "!", "\u2014")) and not is_item):
                lines[-1] = f"{previous} {line}"
                continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_contents(text: str) -> bool:
    """True for a table-of-contents chunk: mostly short numbered headings ("79. Constitution of Parliament.")."""
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    numbered = sum(1 for l in lines if _NUMBERED.match(l) and len(l) < 90)
    return numbered / len(lines) > 0.5


def part_key(section: str) -> str:
    """The top-level part of a section path, without its name: "Part V – The Union › Article 52" and "Part V" are both "Part V"."""
    return (section or "(no heading)").split(" › ")[0].split(" – ")[0].strip()
