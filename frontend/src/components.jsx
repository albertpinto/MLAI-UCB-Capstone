import { useRef, useState } from "react";
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

  return (
    <>
      <div
        className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors hover:border-accent hover:bg-soft ${over ? "border-accent bg-soft text-ink" : "border-line bg-panel text-muted"}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files); }}
        onClick={() => input.current.click()}
        role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current.click()}
      >
        Drop .pdf, .docx or .txt files here, or click to choose
        <input ref={input} type="file" multiple hidden accept=".pdf,.docx,.txt" onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
      </div>
      {children?.(run, running)}
      {running && <p className="hint my-3" aria-live="polite">Working… {items.length ? `${finished} of ${items.length} documents done. ` : ""}Large documents take a while (each chunk is sent to the LLM).</p>}
      {error && <p className="my-3 text-bad" role="alert">{error}</p>}
      {items.length > 0 && (
        <table className="mt-4 w-full overflow-x-auto rounded-xl border border-line bg-panel text-sm">
          <thead className="bg-bg text-left text-xs uppercase tracking-wider text-muted"><tr className="[&>th]:px-3 [&>th]:py-2.5"><th>File</th><th>Status</th><th>Chunks</th><th>Q&amp;A</th><th>Topics</th></tr></thead>
          <tbody>
            {items.map((i) => (
              <tr key={i.name} className="border-t border-line hover:bg-soft [&>td]:px-3 [&>td]:py-2">
                <td>{i.name}</td>
                <td className={i.status === "processed" ? "text-ok" : i.status === "failed" ? "text-bad" : "text-muted"}>{i.status}{i.error ? `: ${i.error}` : ""}</td>
                <td>{i.chunks}</td><td>{i.qa_pairs}</td><td>{i.topics}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
