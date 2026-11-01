import pytest

from capstone.chunking import is_contents, reflow, chunk_text, clean_text
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


def test_reflow_joins_wrapped_lines_but_keeps_list_items():
    text = "The enforcement of any disability arising out\nof Untouchability shall be an offence.\n18. Abolition of titles.\n(1) No title, not being a military\nor academic distinction, shall be conferred by the consent of the\nPresident any title.\nRight to Freedom\n19. Protection of rights.\n(a) to freedom of speech;\n(b) to assemble;"
    assert reflow(text).split("\n") == [
        "The enforcement of any disability arising out of Untouchability shall be an offence.",
        "18. Abolition of titles.",
        "(1) No title, not being a military or academic distinction, shall be conferred by the consent of the President any title.",
        "Right to Freedom",
        "19. Protection of rights.",
        "(a) to freedom of speech;",
        "(b) to assemble;",
    ]
    assert reflow("one\n\ntwo") == "one\n\ntwo"


def test_contents_pages_are_detected():
    contents = "\n".join(f"{n}. Heading number {n}." for n in range(75, 90))
    assert is_contents(contents)
    assert not is_contents("19. Protection of rights.\n(1) All citizens shall have the right to speak freely and to assemble peaceably.\n(2) Nothing in this clause affects the law.")
