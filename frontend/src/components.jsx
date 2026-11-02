import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { ingestFiles, watchJob } from "./api.js";

// Company logo from config; falls back to the first letter of the name if there is no logo or it fails to load.
export function Logo({ branding, size = "size-8" }) {
  const [broken, setBroken] = useState(false);
  if (branding.logo_url && !broken) {
    return <img src={branding.logo_url} alt="" className={`${size} rounded-lg object-contain`} onError={() => setBroken(true)} />;
  }
  return <span className={`${size} grid place-items-center rounded-lg bg-accent font-bold text-white`}>{branding.name.charAt(0)}</span>;
}

// Drag-and-drop / click-to-choose upload with a per-file result table. `children(run, running)` renders extra buttons.
export function UploadBox({ onDone, children }) {
  const [items, setItems] = useState([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [over, setOver] = useState(false);
  const input = useRef();

  async function run(start) {
    setRunning(true);
    setError("");
    setItems([]);
    try {
      const { job_id } = await start();
      await watchJob(job_id, (job) => setItems(job.items));
      onDone?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  const upload = (fileList) => fileList.length && run(() => ingestFiles([...fileList]));
  const finished = items.filter((i) => !["queued", "running"].includes(i.status)).length;
  const current = items.find((i) => i.status === "running");
  // Overall progress: each document counts for the share of its chunks the LLM has finished.
  const fraction = items.length ? items.reduce((sum, i) => sum + (["queued", "running"].includes(i.status) ? (i.chunks ? i.enriched / i.chunks : 0) : 1), 0) / items.length : 0;

  return (
    <>
      <div
        className={`cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-colors hover:border-accent hover:bg-soft ${over ? "border-accent bg-soft" : "border-line bg-panel"}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files); }}
        onClick={() => input.current.click()}
        role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current.click()}
      >
        <UploadCloud size={32} className="mx-auto mb-3 text-accent" strokeWidth={1.5} />
        <p className="font-medium">Drop files here, or click to choose</p>
        <p className="hint mt-1">PDF, DOCX or TXT. Several files at once are fine.</p>
        <input ref={input} type="file" multiple hidden accept=".pdf,.docx,.txt" onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
      </div>
      {children?.(run, running)}
      {running && (
        <div className="my-4" aria-live="polite">
          <div className="h-1.5 overflow-hidden rounded-full bg-line"><div className={`h-full rounded-full bg-linear-to-r from-indigo-600 to-violet-600 transition-all ${items.length ? "" : "w-1/12 animate-pulse"}`} style={items.length ? { width: `${Math.max(3, fraction * 100)}%` } : undefined} /></div>
          <p className="hint mt-2">
            {items.length ? `${finished} of ${items.length} documents done. ` : ""}
            {current ? `${current.name}: ${STAGES[current.stage] ?? "working"}${current.stage === "enriching" ? ` (${current.enriched} of ${current.chunks} chunks)` : ""}. ` : ""}
            Large documents take a while: each section is read and enriched by the LLM.
          </p>
        </div>
      )}
      {error && <p className="my-3 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      {items.length > 0 && (
        <div className="mt-4 overflow-x-auto rounded-2xl border border-line bg-panel shadow-card">
          <table className="w-full text-sm">
            <thead className="bg-bg text-left text-xs uppercase tracking-wider text-muted"><tr className="[&>th]:px-4 [&>th]:py-3"><th>File</th><th>Status</th><th>Sections</th><th>Chunks</th><th>Q&amp;A</th><th>Facts</th><th>Triplets</th><th>Terms</th></tr></thead>
            <tbody>
              {items.map((i) => (
                <tr key={i.name} className="border-t border-line [&>td]:px-4 [&>td]:py-3">
                  <td className="font-medium">{i.name}</td>
                  <td><StatusBadge item={i} /></td>
                  <td>{i.parents}</td><td>{i.chunks}</td><td>{i.qa_pairs}</td><td>{i.factoids}</td><td>{i.triplets}</td><td>{i.topics}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

const STAGES = { reading: "reading the file", splitting: "splitting into sections and chunks", enriching: "enriching chunks with the LLM", summarising: "writing the summary and glossary", storing: "embedding and storing" };

function StatusBadge({ item }) {
  const style = { processed: "bg-ok/15 text-ok", failed: "bg-bad/15 text-bad", running: "bg-soft text-accent", skipped: "bg-line text-muted" }[item.status] ?? "bg-line text-muted";
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${style}`} title={item.error ?? undefined}>{item.status === "running" && item.stage ? item.stage : item.status}{item.error ? `: ${item.error.slice(0, 60)}` : ""}</span>;
}
