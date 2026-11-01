import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, BookOpen, ChevronLeft, ChevronRight } from "lucide-react";
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
  const nav = "inline-flex items-center gap-1.5 rounded-xl border border-line bg-panel px-3.5 py-2 text-sm shadow-sm transition hover:border-accent hover:text-accent";

  return (
    <div className="mx-auto max-w-3xl">
      <button className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted transition hover:text-accent" onClick={() => navigate(-1)}><ArrowLeft size={16} />Back to results</button>
      {error && <p className="rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      {!chunk && !error && <p className="hint">Loading…</p>}
      {chunk && (
        <article className="card !my-0 !p-6 md:!p-8">
          <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-accent"><BookOpen size={14} />{chunk.title}</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">{chunk.section || "Original passage"}</h1>
          <p className="mb-5 mt-1 text-sm text-muted">Passage {chunk.parent_id + 1} of {chunk.total}{chunk.page ? ` · page ${chunk.page}` : ""}</p>
          <pre className="whitespace-pre-wrap break-words border-t border-line pt-5 font-sans text-[15.5px] leading-[1.75]">{chunk.text}</pre>
          <div className="mt-8 flex justify-between border-t border-line pt-4">
            {id > 0 ? <Link className={nav} to={step(id - 1)}><ChevronLeft size={16} />Previous</Link> : <span />}
            {id + 1 < chunk.total ? <Link className={nav} to={step(id + 1)}>Next<ChevronRight size={16} /></Link> : <span />}
          </div>
        </article>
      )}
    </div>
  );
}
