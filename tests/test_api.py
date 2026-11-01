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
    assert {r["type"] for r in body["results"]} <= {"chunk", "qa_pair", "topic", "summary"}

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
