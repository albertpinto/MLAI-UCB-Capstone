import json

from capstone import graph as kg
from capstone.evaluation import Corpus, graph_scorer
from capstone.store import Store

from conftest import FakeEmbedder
from test_api import DOC, ingest


def triplet(s, r, o, chunk=0, document="d1"):
    return {"subject": s, "relation": r, "object": o, "document_id": document, "parent_id": 0, "chunk_id": chunk}


TRIPLETS = [
    triplet("The Governors", "appoint", "Chief Minister", 1),
    triplet("governor", "dismisses", "Minister", 2),
    triplet("Governor's office", "is in", "State capital", 3),
    triplet("Chief Minister", "leads", "Council of Ministers", 4),
]


def test_names_that_differ_in_case_plural_and_article_are_one_entity():
    assert kg.normalise("The Governors") == kg.normalise("governor") == kg.normalise("Governor's") == "governor"
    assert kg.normalise("Article 21") == "article 21" and kg.normalise("Status") == "status"
    assert kg.normalise("SC", {"sc": "supreme court"}) == "supreme court"


def test_the_graph_merges_spellings_and_remembers_each_edge_s_chunk():
    graph = kg.build(TRIPLETS + [TRIPLETS[0]])  # a repeated triplet is one edge
    assert graph.number_of_edges() == 4 and "governor" in graph and graph.nodes["governor"]["label"]
    assert {(d["document_id"], d["chunk_id"]) for _, _, d in graph.out_edges("governor", data=True)} == {("d1", 1), ("d1", 2)}
    assert kg.build([triplet("A", "is", "a")]).number_of_edges() == 0  # an entity related to itself says nothing


def test_save_and_load_round_trip(tmp_path):
    graph = kg.build(TRIPLETS)
    kg.save(graph, tmp_path / "g" / "graph.json")
    loaded = kg.load(tmp_path / "g" / "graph.json")
    assert sorted(loaded.nodes) == sorted(graph.nodes) and loaded.number_of_edges() == 4 and kg.load(tmp_path / "none.json") is None


def test_entities_in_a_question_are_linked_and_lead_to_chunks():
    graph = kg.build(TRIPLETS)
    entities = kg.link_entities(graph, "Who appoints the governors of the chief minister?")
    assert "governor" in entities and "chief minister" in entities
    ranked = kg.graph_chunks(graph, entities, "Who appoints the governors of the chief minister?")
    assert ranked[0][0] == ("d1", 1) and set(ranked[0][1]) == {"Governors", "Chief Minister"}  # the edge between both entities comes first
    assert kg.link_entities(graph, "Nothing relevant here") == []


def test_a_common_entity_leads_to_chunks_only_through_its_relation_or_a_second_entity():
    many = [triplet("President", "signs", f"Bill {i}", i) for i in range(kg.SPECIFIC_DEGREE + 5)] + [triplet("President", "dissolves", "House", 99)]
    graph = kg.build(many)
    assert kg.graph_chunks(graph, ["president"], "Tell me about the President") == []
    assert [c for c, _ in kg.graph_chunks(graph, ["president"], "Can the President dissolve it?")] == [("d1", 99)]


def test_exploring_an_entity():
    graph = kg.build(TRIPLETS)
    assert kg.search_entities(graph, "Minister")[0]["id"] in {"minister", "chief minister"}
    near = kg.neighbourhood(graph, "chief minister")
    assert near["nodes"][0]["centre"] and {n["id"] for n in near["nodes"]} == {"chief minister", "governor", "council of minister"}
    assert {f["relation"] for f in kg.facts(graph, "governor")} == {"appoint", "dismisses"}
    stats = kg.stats(graph)
    assert stats["entities"] == graph.number_of_nodes() and stats["documents"] == 1
    assert kg.summary_text(graph, "Guide", "d1").startswith("Knowledge graph of Guide: 6 entities and 4 relations")
    assert kg.summary_text(graph, "Guide", "other") == ""


def stored_chunks():
    store = Store(":memory:", "g", FakeEmbedder())
    texts = ["Governors appoint the chief minister of a state", "Ministers form the council", "Courts hear cases about disputes", "Taxes are collected by the union",
             "Elections are held every five years", "Rivers flow to the sea"]
    base = {"document_id": "d1", "title": "T", "citation": "T", "file_hash": "h", "parent_id": 0, "section": "S", "page": 1, "parent_text": "p", "parents_total": 1, "topics": []}
    store.add([(t, {**base, "type": "chunk", "chunk_id": i + 1, "content": t}) for i, t in enumerate(texts)])
    return store


def test_search_adds_the_chunks_the_graph_points_at(monkeypatch):
    store = stored_chunks()
    question = "who appoints the governor"
    plain = store.search(question, 3, ["chunk"])
    left_out = min({1, 2, 3, 4, 5, 6} - {r["chunk_id"] for r in plain})  # a chunk the dense search did not return
    graph = kg.build([triplet("Governor", "appoints", "Chief Minister", left_out)])  # the graph ties the governor to it
    assert [r["chunk_id"] for r in store.search(question, 3, ["chunk"], graph=graph)] == [r["chunk_id"] for r in plain]  # at the default weight the graph only nudges
    monkeypatch.setattr(kg, "GRAPH_WEIGHT", 1.0)
    assisted = store.search(question, 3, ["chunk"], graph=graph)
    assert not any("graph" in r for r in plain)
    added = next(r for r in assisted if r["chunk_id"] == left_out)  # fetched from the graph, with a cosine score like any result
    assert added["graph"] == ["Governor"] and isinstance(added["score"], float) and added["text"]


def test_the_evaluation_scorer_boosts_graph_chunks():
    graph = kg.build([triplet("Governor", "appoints", "Chief Minister", 3)])
    corpus = Corpus(["a", "b", "c"], ["s0", "s1", "s2"], {}, [("d1", 1), ("d1", 2), ("d1", 3)])
    score = graph_scorer(lambda q: [0.9, 0.5, 0.1], graph, corpus)
    assert score("nothing here").argmax() == 0  # no entity in the question: the dense order stands
    assert score("Who appoints the Governor?").argmax() == 2  # first in the graph and third in the dense order beats first in the dense order alone


def test_graph_endpoints_and_rebuilds(client, settings):
    assert client.get("/api/graph/stats").json() == {"built": False}
    assert client.get("/api/graph/entity", params={"id": "x"}).status_code == 404
    ingest(client, "Guide", DOC)  # the job rebuilds the graph from the new triplets
    assert settings.graph_path.is_file()
    stats = client.get("/api/graph/stats").json()
    assert stats["built"] and stats["entities"] >= 2 and stats["documents"] == 1
    found = client.get("/api/graph/entities", params={"q": "add brand"}).json()["entities"]
    assert found and found[0]["id"] == "add brand button"
    entity = client.get("/api/graph/entity", params={"id": found[0]["id"]}).json()
    assert entity["nodes"][0]["centre"] and entity["facts"][0]["relation"] == "creates" and entity["facts"][0]["text"]
    summaries = client.post("/api/search", json={"query": "knowledge graph", "types": ["graph_summary"]}).json()["results"]
    assert len(summaries) == 1 and summaries[0]["content"].startswith("Knowledge graph of Guide")
    assert client.post("/api/graph/rebuild").json()["entities"] == stats["entities"]
    assert len(client.post("/api/search", json={"query": "knowledge graph", "types": ["graph_summary"]}).json()["results"]) == 1  # replaced, not added
    on = client.post("/api/search", json={"query": "What does the Add Brand button do?", "graph": True}).json()["results"]
    assert any(r.get("graph") for r in on)
    assert not any("graph" in r for r in client.post("/api/search", json={"query": "What does the Add Brand button do?"}).json()["results"])
    doc_id = client.get("/api/documents").json()["documents"][0]["document_id"]
    client.delete(f"/api/documents/{doc_id}")
    assert client.get("/api/graph/stats").json()["entities"] == 0
