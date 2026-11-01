import { useEffect, useState } from "react";
import { ClipboardPaste, FileText, FolderInput, Library, Trash2, UploadCloud } from "lucide-react";
import { deleteDocument, ingestFolder, ingestText, listDocuments, watchJob } from "../api.js";
import { UploadBox } from "../components.jsx";

const TABS = [
  ["upload", "Upload", UploadCloud],
  ["paste", "Paste text", ClipboardPaste],
  ["library", "Library", Library],
];

export default function Documents() {
  const [tab, setTab] = useState("upload");
  const [docs, setDocs] = useState([]);
  const [error, setError] = useState("");

  const refresh = () => listDocuments().then((d) => setDocs(d.documents)).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
      <p className="hint mb-5 mt-1">Add documents to the knowledge base, or manage what is already in it.</p>
      <div className="mb-5 flex gap-1 rounded-xl border border-line bg-panel p-1" role="tablist">
        {TABS.map(([id, label, Icon]) => (
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
            className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm transition ${tab === id ? "bg-soft font-medium text-accent" : "text-muted hover:text-ink"}`}>
            <Icon size={16} />{label}{id === "library" && <span className="rounded-full bg-bg px-1.5 text-xs">{docs.length}</span>}
          </button>
        ))}
      </div>
      {error && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      {tab === "upload" && <Upload onDone={refresh} />}
      {tab === "paste" && <Paste onDone={refresh} />}
      {tab === "library" && <LibraryTab docs={docs} refresh={refresh} setError={setError} />}
    </div>
  );
}

function Upload({ onDone }) {
  return (
    <UploadBox onDone={onDone}>
      {(run, running) => (
        <div className="my-4 flex flex-wrap items-center gap-3">
          <button className="btn inline-flex items-center gap-2" disabled={running} onClick={() => run(ingestFolder)}><FolderInput size={16} />Ingest the docs folder</button>
          <span className="hint">Reads every file in the server's DOCS_DIR.</span>
        </div>
      )}
    </UploadBox>
  );
}

function Paste({ onDone }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const { job_id } = await ingestText(title, text);
      const item = (await watchJob(job_id, () => {})).items[0];
      if (item.status === "failed") throw new Error(item.error);
      setMessage(item.status === "skipped" ? "That text is already stored." : `Added: ${item.chunks} chunks, ${item.qa_pairs} Q&A pairs, ${item.topics} topics.`);
      if (item.status === "processed") { setTitle(""); setText(""); onDone(); }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3">
      <input className="field" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title (used as the citation)" required />
      <textarea className="field" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the document text here" rows={12} required />
      <button className="btn" disabled={busy}>{busy ? "Adding…" : "Add document"}</button>
      {message && <p className="text-ok" role="status">{message}</p>}
      {error && <p className="text-bad" role="alert">{error}</p>}
    </form>
  );
}

function LibraryTab({ docs, refresh, setError }) {
  const [filter, setFilter] = useState("");
  const shown = docs.filter((d) => d.title.toLowerCase().includes(filter.trim().toLowerCase()));

  async function remove(doc) {
    if (!window.confirm(`Delete "${doc.title}" and everything derived from it?`)) return;
    await deleteDocument(doc.document_id).catch((e) => setError(e.message));
    refresh();
  }

  return (
    <>
      <input className="field mb-3" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={`Filter ${docs.length} documents`} aria-label="Filter documents" />
      {docs.length === 0 && <p className="hint rounded-2xl border border-dashed border-line p-8 text-center">No documents yet. Upload some on the first tab.</p>}
      {shown.length > 0 && (
        <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel shadow-card">
          {shown.map((d) => (
            <li key={d.document_id} className="flex items-center gap-3 px-4 py-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-soft text-accent"><FileText size={18} /></span>
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{d.title}</span>
                <span className="block truncate text-xs text-muted">{d.document_id}</span>
              </span>
              <button onClick={() => remove(d)} className="rounded-lg p-2 text-muted transition hover:bg-bad/10 hover:text-bad" aria-label={`Delete ${d.title}`}><Trash2 size={16} /></button>
            </li>
          ))}
        </ul>
      )}
      {docs.length > 0 && shown.length === 0 && <p className="hint">No documents match "{filter}".</p>}
    </>
  );
}
