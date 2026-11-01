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
