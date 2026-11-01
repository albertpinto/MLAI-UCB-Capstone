import pytest

from capstone.chunking import chunk_text, clean_text
from capstone.extract import extract_text


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_chunks_respect_size_and_cover_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 50 + 2 for c in chunks)
    assert "Paragraph 0." in chunks[0] and "Paragraph 19." in chunks[-1]


def test_long_paragraph_is_split():
    chunks = chunk_text("Sentence one. " * 200, size=300, overlap=0)
    assert len(chunks) > 5 and all(len(c) <= 300 for c in chunks)


def test_extract_txt_and_unsupported():
    assert extract_text("a.txt", b"hello") == "hello"
    with pytest.raises(ValueError):
        extract_text("a.xyz", b"")
