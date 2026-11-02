import time

DOC = "Adding a brand\n\nTo add a brand click the Add Brand button.\n\nBrands appear in the dashboard."


def wait(client, job_id):
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def ingest(client, title="Guide", text=DOC):
    r = client.post("/api/ingest/text", json={"title": title, "text": text})
    assert r.status_code == 200
    return wait(client, r.json()["job_id"])["items"][0]


def test_ingest_search_and_dedup(client):
    item = ingest(client)
    assert item["status"] == "processed" and item["chunks"] >= 1 and item["qa_pairs"] >= 1
    assert ingest(client)["status"] == "skipped"

    body = client.post("/api/search", json={"query": "how do I add a brand?", "limit": 5}).json()
    assert body["results"] and body["answer"]["citation"] == "Guide"
    assert {r["type"] for r in body["results"]} <= {"chunk", "qa_pair", "factoid", "topic", "summary"}  # triplets only when asked for

    only_topics = client.post("/api/search", json={"query": "brand", "types": ["topic"]}).json()["results"]
    assert only_topics and all(r["type"] == "topic" for r in only_topics)
    assert client.get("/api/topics").json()["topics"][0]["term"] == "brand"


def test_documents_list_and_delete(client):
    ingest(client)
    docs = client.get("/api/documents").json()["documents"]
    assert [d["title"] for d in docs] == ["Guide"]
    client.delete(f"/api/documents/{docs[0]['document_id']}")
    assert client.get("/api/documents").json()["documents"] == []
    assert client.post("/api/search", json={"query": "brand"}).json()["results"] == []


def test_file_upload_and_validation(client):
    r = client.post("/api/ingest/files", files=[("files", ("guide.txt", DOC.encode(), "text/plain"))])
    assert wait(client, r.json()["job_id"])["items"][0]["status"] == "processed"
    assert client.post("/api/ingest/files", files=[("files", ("bad.exe", b"x"))]).status_code == 400
    assert client.post("/api/ingest/text", json={"title": " ", "text": "x"}).status_code == 400


def test_folder_ingest(client, settings):
    (settings.docs_dir / "a.txt").write_text(DOC)
    r = client.post("/api/ingest/folder")
    assert wait(client, r.json()["job_id"])["items"][0]["name"] == "a.txt"


def test_branding_comes_from_settings(client, settings):
    assert client.get("/api/branding").json() == {"name": settings.company_name, "tagline": settings.company_tagline, "logo_url": settings.logo_url}


def test_every_endpoint_is_documented(client):
    schema = client.get("/api/openapi.json").json()
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            assert op.get("summary") and op.get("description") and op.get("tags"), f"{method.upper()} {path} lacks docs"


def test_suggestions_come_from_the_configured_file(client, settings, tmp_path):
    settings.suggested_questions_file = tmp_path / "none.json"
    assert client.get("/api/suggestions").json() == {"suggestions": []}
    settings.suggested_questions_file.write_text('[{"topic": "T", "question": "Q?"}]')
    assert client.get("/api/suggestions").json() == {"suggestions": [{"topic": "T", "question": "Q?"}]}


def test_passage_endpoint_and_contents_pages_are_hidden(client):
    item = ingest(client, "Guide", DOC)
    passage = client.get(f"/api/documents/{item['document_id']}/passages/0").json()
    assert passage["title"] == "Guide" and passage["total"] == item["parents"] and passage["text"] == DOC  # the original text, not the LLM's
    assert client.get(f"/api/documents/{item['document_id']}/passages/99").status_code == 404
    toc = "\n".join(f"{n}. Heading number {n}." for n in range(1, 12))
    ingest(client, "Contents", toc)
    found = client.post("/api/search", json={"query": "Heading number", "limit": 20}).json()["results"]
    assert not [r for r in found if r["title"] == "Contents" and r["type"] == "chunk"]


def test_several_files_in_one_job_are_all_processed_in_order(client):
    files = [("files", (f"g{i}.txt", f"Guide {i}\n\nTo add brand {i} click the Add Brand button.".encode(), "text/plain")) for i in range(3)]
    job = wait(client, client.post("/api/ingest/files", files=files).json()["job_id"])
    assert [(i["name"], i["status"]) for i in job["items"]] == [(f"g{n}.txt", "processed") for n in range(3)]


def test_changed_text_under_the_same_title_replaces_the_document_with_a_new_version(client):
    first = ingest(client, "Guide", DOC)
    second = ingest(client, "Guide", DOC + "\n\nBrands can be archived.")
    assert (first["version"], second["version"], second["status"]) == (1, 2, "processed")
    docs = client.get("/api/documents").json()["documents"]
    assert [(d["title"], d["version"]) for d in docs] == [("Guide", 2)] and docs[0]["document_id"] == second["document_id"]
    assert client.post("/api/search", json={"query": "archived"}).json()["results"][0]["document_id"] == second["document_id"]
    assert ingest(client, "Other", "Passwords\n\nReset them by email.")["version"] == 1  # another title is a new document, not an update


def test_expired_documents_are_flushed_and_others_stay(client, settings, store):
    settings.document_ttl_days = 1
    ingest(client, "Short lived", DOC)
    settings.document_ttl_days = 0
    ingest(client, "Forever", "Passwords\n\nReset them by email.")
    docs = {d["title"]: d for d in client.get("/api/documents").json()["documents"]}
    assert docs["Short lived"]["expires_at"] and docs["Forever"]["expires_at"] is None
    assert client.post("/api/documents/flush").json() == {"flushed": []}  # not due yet
    from unittest.mock import patch
    with patch("capstone.main.time.time", return_value=docs["Short lived"]["expires_at"] + 1):
        flushed = client.post("/api/documents/flush").json()["flushed"]
    assert [f["title"] for f in flushed] == ["Short lived"]
    assert [d["title"] for d in client.get("/api/documents").json()["documents"]] == ["Forever"]


def test_long_documents_are_summarised_from_slices_spread_over_the_whole_text():
    from capstone.ingest import DOC_INPUT_CHARS, _sample

    text = "start " * 10000 + "middle " * 10000 + "end " * 10000
    sample = _sample(text)
    assert len(sample) < DOC_INPUT_CHARS + 50 and "start" in sample and "middle" in sample and "end" in sample
    assert _sample("short") == "short"


def test_the_index_can_live_in_a_folder_without_a_server(tmp_path, store):
    from qdrant_client import QdrantClient

    from capstone.export_index import export_index
    from capstone.store import Store
    from conftest import FakeEmbedder

    store.add([("how do I add a brand", {"type": "chunk", "document_id": "d", "title": "T", "citation": "T", "file_hash": "h", "topics": [], "parent_id": 0, "chunk_id": 0, "content": "how do I add a brand", "section": "", "page": 1})])
    assert export_index(store.client, "test", tmp_path / "q") == 1
    local = Store(tmp_path / "q", "test", FakeEmbedder())  # a Path opens the data folder
    assert local.search("add a brand", limit=3)[0]["content"] == "how do I add a brand"
