import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getPassage } from "../api.js";

// One passage of a document, read in full: where search results and answer sources link to.
export default function Passage() {
  const { documentId, parentId } = useParams();
  const [chunk, setChunk] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const id = Number(parentId);

  useEffect(() => {
    setChunk(null);
    setError("");
    getPassage(documentId, id).then(setChunk).catch((e) => setError(e.message));
  }, [documentId, id]);

  const step = (n) => `/passage/${documentId}/${n}`;
  const nav = "rounded-lg border border-line bg-panel px-3 py-1.5 text-sm hover:border-accent";

  return (
    <>
      <button className="linkbtn mb-3 !px-0" onClick={() => navigate(-1)}>← Back to results</button>
      {error && <p className="text-bad" role="alert">{error}</p>}
      {!chunk && !error && <p className="hint">Loading…</p>}
      {chunk && (
        <article className="card">
          <p className="text-xs font-semibold uppercase tracking-wider text-accent">{chunk.title}</p>
          <h1 className="text-lg font-semibold">{chunk.section || "Original passage"}</h1>
          <p className="mb-3 text-sm text-muted">Passage {chunk.parent_id + 1} of {chunk.total}{chunk.page ? ` · page ${chunk.page}` : ""}</p>
          <pre className="whitespace-pre-wrap break-words font-sans leading-relaxed">{chunk.text}</pre>
          <div className="mt-5 flex justify-between border-t border-line pt-3">
            {id > 0 ? <Link className={nav} to={step(id - 1)}>← Previous passage</Link> : <span />}
            {id + 1 < chunk.total ? <Link className={nav} to={step(id + 1)}>Next passage →</Link> : <span />}
          </div>
        </article>
      )}
    </>
  );
}
