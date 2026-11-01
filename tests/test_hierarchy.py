from types import SimpleNamespace

import pytest

from capstone import parsing
from capstone.config import Settings
from capstone.enrich import EnrichmentCache, enrich_child, grounded, resolved_ok
from capstone.ingest import ingest_text
from capstone.parsing import Parent, garbled, parents_from_file, parents_from_text, split_children

SETTINGS = Settings(qdrant_url=":memory:", parent_chars=300, child_words=40, child_min_words=8, enrich_cache=None)


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
    text = "Adding a brand\nTo add a brand click the Add Brand button.\n\nBrands appear in the dashboard after saving."
    parents = [Parent("To add a brand click the Add Brand button.\n\nBrands appear in the dashboard.", "Brands › Adding", 3)]
    result = ingest_text(store, llm, SETTINGS, "Guide", text, parents=parents)
    assert (result.parents, result.chunks) == (1, 2)
    assert (result.qa_pairs, result.factoids, result.triplets) == (2, 2, 2)  # one of each per child, from the fake LLM

    chunks = [r for r in store.search("brand", 20) if r["type"] == "chunk"]
    assert {c["parent_id"] for c in chunks} == {0} and {c["section"] for c in chunks} == {"Brands › Adding"}
    assert all("parent_text" not in c for c in chunks)  # parents are not repeated in every result
    for kind in ("qa_pair", "factoid"):
        assert all(r["parent_id"] == 0 for r in store.search("brand", 20, [kind]))


def test_references_are_resolved_and_the_original_is_kept(store):
    class ResolvingLLM:
        def json_call(self, system, user, retries=3):
            if "resolve references" in system:
                child = user.split("CHILD:\n", 1)[1]
                return {"replacements": [{"find": "It shows", "replace": "The dashboard shows"}], "resolved": child.replace("It shows", "The dashboard shows")}
            if "search index" in system:
                return {"qa_pairs": [], "factoids": [], "triplets": []}
            return {"topics": [], "summary": ""}

    parents = [Parent("The dashboard lists brands.\n\nIt shows one row per brand.")]
    ingest_text(store, ResolvingLLM(), SETTINGS, "Guide", "x", parents=parents)
    chunks = [r for r in store.search("dashboard shows", 10, ["chunk"])]
    assert any(c["content"] == "The dashboard shows one row per brand." and c["original"] == "It shows one row per brand." for c in chunks)


def test_triplets_are_searchable_only_when_asked_for(store, llm):
    ingest_text(store, llm, SETTINGS, "Guide", "x", parents=[Parent("To add a brand click the Add Brand button.", "S", 1)])
    assert not [r for r in store.search("Add Brand button creates a brand", 20) if r["type"] == "triplet"]
    found = store.search("Add Brand button creates a brand", 20, ["triplet"])
    assert found and found[0]["text"] == "Add Brand button → creates → a brand" and found[0]["subject"] == "Add Brand button"


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
            if "resolve references" in system:
                return {"resolved": ["not", "a", "string"]}  # wrong type, but still an object
            if "search index" in system:
                return {"qa_pairs": "nope", "factoids": {"a": 1}, "triplets": None}
            return {"topics": [], "summary": ""}

    parents = [Parent("Click Add Brand.", "S", 1)]
    result = ingest_text(store, OddLLM(), SETTINGS, "Guide", "Click Add Brand.", parents=parents)
    assert result.status == "processed" and result.qa_pairs == 0
    assert [r["content"] for r in store.search("brand", 5) if r["type"] == "chunk"] == ["Click Add Brand."]  # raw child kept


def test_an_enrichment_call_that_raises_keeps_the_raw_child(store):
    class DownLLM:
        def json_call(self, system, user, retries=3):
            if "resolve references" in system or "search index" in system:
                raise RuntimeError("rate limited")
            return {"topics": [], "summary": ""}

    result = ingest_text(store, DownLLM(), SETTINGS, "Guide", "Click Add Brand.", parents=[Parent("Click Add Brand.")])
    assert result.status == "processed" and result.chunks == 1


def test_contents_pages_are_not_made_into_parents(monkeypatch):
    toc = [item("text", f"{n}. Heading number {n}.", page=5) for n in range(75, 125)]
    with_docling(monkeypatch, [item("section_header", "Contents")] + toc + [item("section_header", "Part I"), item("text", "The Union shall be a Federation of States. " * 12, page=9)])
    parents = parents_from_file("c.pdf", b"x", Settings(qdrant_url=":memory:", parent_chars=2000))  # a realistic parent size
    assert [p.page for p in parents] == [9] and "Heading number" not in parents[0].text


def test_resolved_text_must_keep_the_numbers_and_stay_about_the_same_length():
    original = "Article 85: he may summon each House within 6 months."
    assert resolved_ok(original, "Article 85: the President may summon each House within 6 months.")
    assert not resolved_ok(original, "Article 85: the President may summon each House within 7 months.")  # a number changed
    assert not resolved_ok(original, "Article 85: the President may summon each House within 6 months, and the President may also do many other things besides that.")  # ballooned
    assert not resolved_ok(original, "") and not resolved_ok(original, ["not text"])


def test_invented_facts_and_vague_questions_are_dropped():
    source = "The President shall summon each House of Parliament to meet at such time and place as the President thinks fit."
    assert grounded("The President summons each House of Parliament.", source)
    assert not grounded("The Prime Minister dissolves the Supreme Court.", source)
    reply = {
        "resolved": source,
        "qa_pairs": [{"question": "Who summons each House?", "answer": "The President."}, {"question": "What does the text say about summoning?", "answer": "x"}, {"question": "Q?"}],
        "factoids": ["The President summons each House of Parliament.", "The Prime Minister dissolves the Supreme Court.", 42, "This provision summons each House of Parliament.", "He summons each House of Parliament."],
        "triplets": [
            {"subject": "The President", "relation": "summons", "object": "each House of Parliament"},
            {"subject": "The Governor", "relation": "summons", "object": "each House of Parliament"},  # subject never appears
            {"subject": "The President", "relation": "summons", "object": ""},
        ],
    }
    e = enrich_child(ScriptedLLM(reply), SETTINGS, Parent(source), source)
    assert [p["question"] for p in e.qa_pairs] == ["Who summons each House?"]
    assert e.factoids == ["The President summons each House of Parliament."]
    assert e.triplets == [{"subject": "The President", "relation": "summons", "object": "each House of Parliament"}]


def test_a_reply_with_the_wrong_shape_keeps_the_original_child():
    e = enrich_child(ScriptedLLM({"resolved": 7, "qa_pairs": "no", "factoids": {"a": 1}, "triplets": None}), SETTINGS, Parent("Some text here."), "Some text here.")
    assert (e.content, e.qa_pairs, e.factoids, e.triplets) == ("Some text here.", [], [], [])


def test_the_cache_saves_llm_calls_and_survives_a_restart(tmp_path):
    reply = {"resolved": "The President acts.", "qa_pairs": [], "factoids": [], "triplets": []}
    llm, parent, path = ScriptedLLM(reply), Parent("The President acts."), tmp_path / "cache" / "e.jsonl"
    enrich_child(llm, SETTINGS, parent, "The President acts.", EnrichmentCache(path))
    assert llm.calls == 2  # resolve, then derive
    enrich_child(llm, SETTINGS, parent, "The President acts.", EnrichmentCache(path))  # a new cache object reads the file
    assert llm.calls == 2
    path.write_text(path.read_text() + '{"key": "half-writt')  # a crash mid-write must not break the next run
    enrich_child(llm, SETTINGS, parent, "The President acts.", EnrichmentCache(path))
    assert llm.calls == 2
    enrich_child(llm, SETTINGS, parent, "Another child.", EnrichmentCache(path))
    assert llm.calls == 3  # a different child needs a new resolve call (its derive call hits the cache: the scripted reply resolves to the same text)


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def json_call(self, system, user, retries=3):
        self.calls += 1
        return self.reply


def test_running_page_headers_name_the_part_and_do_not_cut_parents(monkeypatch):
    body = "Every citizen has the right to speak freely and to assemble peaceably in public. " * 4
    items = []
    for page in range(1, 101):  # 100 pages: Part III on pages 1-25, Part IV on 26-50, then no part header
        items.append(item("section_header", "THE CONSTITUTION OF INDIA", page=page))  # on every page: the document title
        if page <= 25:
            items.append(item("section_header", "(Part III.-Fundamental Rights)", page=page))
        elif page <= 50:
            items.append(item("section_header", "(Part IV.-Directive Principles)", page=page))
        items.append(item("text", body, page=page))
    items.insert(5, item("section_header", "Right to Freedom", page=3, level=1))  # a real, one-off heading
    with_docling(monkeypatch, items)
    parents = parents_from_file("c.pdf", b"x", Settings(qdrant_url=":memory:", parent_chars=700))
    sections = {p.section for p in parents}
    assert "Part III – Fundamental Rights" in sections and "Part IV – Directive Principles" in sections
    assert any(s == "Part III – Fundamental Rights › Right to Freedom" for s in sections)
    assert not any("THE CONSTITUTION" in s for s in sections)
    assert not any(p.page > 25 and "Part III" in p.section for p in parents)  # a new part always starts a new parent


def test_a_very_long_child_is_cut_into_pieces():
    from capstone.parsing import _cap

    long = "; ".join(f"clause {i} says something specific about item {i}" for i in range(40))
    pieces = _cap(long, 60, 30)
    assert len(pieces) > 5 and all(len(p.split()) <= 45 for p in pieces) and " ".join(pieces).count("clause") == 40
    assert _cap("short enough text", 60, 30) == ["short enough text"]
