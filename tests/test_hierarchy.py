from types import SimpleNamespace

import pytest

from capstone import parsing
from capstone.config import Settings
from capstone.ingest import ingest_text
from capstone.parsing import Parent, garbled, parents_from_file, parents_from_text, split_children

SETTINGS = Settings(qdrant_url=":memory:", parent_chars=300, child_words=40, child_min_words=8)


def item(label, text, page=1, level=1):
    return (SimpleNamespace(label=SimpleNamespace(value=label), text=text, prov=[SimpleNamespace(page_no=page)]), level)


class FakeConverter:
    def __init__(self, items):
        self.items = items

    def convert(self, source):
        return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: iter(self.items)))


def with_docling(monkeypatch, items):
    monkeypatch.setattr(parsing, "_converter", lambda threads: FakeConverter(items))


def test_text_parents_pack_whole_paragraphs():
    parents = parents_from_text("\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(10)), 300)
    assert len(parents) > 1 and all(len(p.text) <= 330 for p in parents)


def test_docling_parents_follow_headings_and_drop_page_furniture(monkeypatch):
    body = "Every citizen has the right to speak freely and to assemble peaceably. " * 8
    with_docling(monkeypatch, [
        item("page_header", "PART III"),
        item("section_header", "Fundamental Rights", level=1),
        item("text", body, page=7),
        item("footnote", "1. Subs. by the Constitution (Forty-fourth Amendment) Act, 1978", page=7),
        item("text", "______________________________________________", page=7),
        item("section_header", "Right to Freedom", level=2),
        item("text", body, page=8),
    ])
    parents = parents_from_file("c.pdf", b"x", SETTINGS)
    assert [(p.section, p.page) for p in parents] == [("Fundamental Rights", 7), ("Fundamental Rights › Right to Freedom", 8)]
    assert not any("Subs. by" in p.text or "____" in p.text or "PART III" in p.text for p in parents)


def test_small_sections_are_merged_and_garbled_text_is_dropped(monkeypatch):
    with_docling(monkeypatch, [
        item("section_header", "A"), item("text", "Short one."),
        item("section_header", "B"), item("text", "Short two."),
        item("text", "£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ " * 20),
    ])
    parents = parents_from_file("c.docx", b"x", SETTINGS)
    assert len(parents) == 1 and "Short one." in parents[0].text and "Short two." in parents[0].text
    assert garbled("£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ") and not garbled("The Constitution of India")


def test_docling_failure_falls_back_to_plain_text(monkeypatch):
    def boom(threads):
        raise RuntimeError("no model")

    monkeypatch.setattr(parsing, "_converter", boom)
    assert parents_from_file("a.txt", b"hello", SETTINGS)[0].text == "hello"  # .txt never needs Docling
    monkeypatch.setattr(parsing, "extract_text", lambda name, data: "plain fallback text")
    assert parents_from_file("a.pdf", b"x", SETTINGS)[0].text == "plain fallback text"


def test_split_children_with_the_real_chunker_keeps_pieces_big_enough():
    try:
        children = split_children(
            "Article 14 says the State shall not deny equality before the law. Article 15 forbids discrimination on grounds of religion, race, caste, sex or place of birth. "
            "Article 16 gives equality of opportunity in public employment. The sea is salty and the sky is blue. Penguins live in cold places and eat fish. "
            "Bananas are yellow when ripe and green when they are not. " * 2,
            Settings(child_words=30, child_min_words=10),
        )
    except Exception as exc:  # the embedding model is not cached and there is no network
        pytest.skip(f"chunker model unavailable: {exc}")
    assert len(children) > 1 and all(len(c.split()) >= 10 for c in children[:-1])


def test_children_are_linked_to_parents_and_enriched(store, llm):
    text = "Adding a brand\\nTo add a brand click the Add Brand button.\\n\\n".replace("\\n", "\n") + "Brands appear in the dashboard after saving."
    parents = [Parent("To add a brand click the Add Brand button.\n\nBrands appear in the dashboard.", "Brands › Adding", 3)]
    result = ingest_text(store, llm, SETTINGS, "Guide", text, parents=parents)
    assert (result.parents, result.chunks) == (1, 2)

    chunks = [r for r in store.search("brand", 20) if r["type"] == "chunk"]
    assert {c["parent_id"] for c in chunks} == {0} and {c["section"] for c in chunks} == {"Brands › Adding"}
    assert all(c["content"].startswith("About brands.") for c in chunks)  # the anaphora step's context was prepended
    assert all("parent_text" not in c for c in chunks)  # parents are not repeated in every result
    assert all(r["parent_id"] == 0 for r in store.search("brand", 20) if r["type"] == "qa_pair")


def test_small_to_big_returns_each_parent_once_with_its_full_text(store, llm):
    parents = [Parent("Click Add Brand.\n\nBrands appear in the dashboard.", "S1", 1), Parent("Delete a brand from settings.\n\nIt is removed.", "S2", 2)]
    ingest_text(store, llm, SETTINGS, "Guide", "x " + "Click Add Brand. Delete a brand from settings.", parents=parents)
    results = store.with_parents(store.search("brand", 30))
    contexts = [r["context"] for r in results if "context" in r]
    assert sorted(contexts) == sorted(p.text for p in parents)  # whole sections, one each, though many children and Q&As matched
    assert store.passage(results[0]["document_id"], 1)["section"] == "S2"


def test_a_malformed_llm_reply_for_one_child_does_not_abort_the_document(store):
    class OddLLM:
        def json_call(self, system, user, retries=3):
            if "context_passage" in system:
                return {"context_passage": ["not", "a", "string"], "qa_pairs": "nope"}  # wrong types, but still an object
            return {"topics": [], "summary": ""}

    parents = [Parent("Click Add Brand.", "S", 1)]
    result = ingest_text(store, OddLLM(), SETTINGS, "Guide", "Click Add Brand.", parents=parents)
    assert result.status == "processed" and result.qa_pairs == 0
    assert [r["content"] for r in store.search("brand", 5) if r["type"] == "chunk"] == ["Click Add Brand."]  # raw child kept


def test_an_enrichment_call_that_raises_keeps_the_raw_child(store):
    class DownLLM:
        def json_call(self, system, user, retries=3):
            if "context_passage" in system:
                raise RuntimeError("rate limited")
            return {"topics": [], "summary": ""}

    result = ingest_text(store, DownLLM(), SETTINGS, "Guide", "Click Add Brand.", parents=[Parent("Click Add Brand.")])
    assert result.status == "processed" and result.chunks == 1


def test_contents_pages_are_not_made_into_parents(monkeypatch):
    toc = [item("text", f"{n}. Heading number {n}.", page=5) for n in range(75, 125)]
    with_docling(monkeypatch, [item("section_header", "Contents")] + toc + [item("section_header", "Part I"), item("text", "The Union shall be a Federation of States. " * 12, page=9)])
    parents = parents_from_file("c.pdf", b"x", Settings(qdrant_url=":memory:", parent_chars=2000))  # a realistic parent size
    assert [p.page for p in parents] == [9] and "Heading number" not in parents[0].text
