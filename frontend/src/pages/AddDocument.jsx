import { useEffect, useState } from "react";
import { UploadBox } from "../components.jsx";
import { deleteDocument, ingestText, listDocuments, watchJob } from "../api.js";

export default function AddDocument() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [docs, setDocs] = useState([]);

  const refresh = () => listDocuments().then((d) => setDocs(d.documents)).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const { job_id } = await ingestText(title, text);
      const job = await watchJob(job_id, () => {});
      const item = job.items[0];
      if (item.status === "failed") throw new Error(item.error);
      setMessage(item.status === "skipped" ? "That text is already stored." : `Added: ${item.chunks} chunks, ${item.qa_pairs} Q&A pairs, ${item.topics} topics.`);
      if (item.status === "processed") { setTitle(""); setText(""); }
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(doc) {
    if (!window.confirm(`Delete "${doc.title}" and everything derived from it?`)) return;
    await deleteDocument(doc.document_id).catch((e) => setError(e.message));
    refresh();
  }

  return (
    <>
      <h1 className="mb-4 text-2xl font-semibold tracking-tight">Add documents</h1>
      <h2 className="mb-2 font-semibold">Upload files</h2>
      <UploadBox onDone={refresh} />

      <h2 className="mb-2 mt-8 font-semibold">Or paste text</h2>
      <form onSubmit={submit} className="flex flex-col gap-3">
        <input className="field" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title (used as the citation)" required />
        <textarea className="field" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the document text here" rows={10} required />
        <div><button className="btn" disabled={busy}>{busy ? "Adding…" : "Add document"}</button></div>
      </form>
      {message && <p className="mt-3 text-ok" role="status">{message}</p>}
      {error && <p className="mt-3 text-bad" role="alert">{error}</p>}

      <h2 className="mb-2 mt-8 font-semibold">Stored documents ({docs.length})</h2>
      {docs.length === 0 && <p className="hint">No documents stored yet.</p>}
      <ul className="divide-y divide-line rounded-xl border border-line bg-panel">
        {docs.map((d) => (
          <li key={d.document_id} className="flex items-center justify-between px-4 py-2.5">{d.title} <button className="linkbtn !text-bad" onClick={() => remove(d)}>delete</button></li>
        ))}
      </ul>
    </>
  );
}
