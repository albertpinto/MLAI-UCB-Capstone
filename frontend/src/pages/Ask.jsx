import { useState } from "react";
import { ask as askApi, search } from "../api.js";

const TYPES = [
  ["qa_pair", "Q&A"],
  ["chunk", "Passages"],
  ["summary", "Summaries"],
  ["topic", "Topics"],
];

export default function Ask() {
  const [query, setQuery] = useState("");
  const [types, setTypes] = useState([]);
  const [data, setData] = useState(null);
  const [grounded, setGrounded] = useState(null);
  const [answerError, setAnswerError] = useState("");
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const toggle = (t) => setTypes((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  async function ask(q = query) {
    if (!q.trim()) return;
    setQuery(q);
    setBusy(true);
    setError("");
    try {
      // the grounded answer and the plain results are independent: results still show if the LLM call fails
      const [found, answer] = await Promise.allSettled([search(q, types), askApi(q, types)]);
      if (found.status === "rejected") throw found.reason;
      setData(found.value);
      setGrounded(answer.status === "fulfilled" ? answer.value : null);
      setAnswerError(answer.status === "rejected" ? answer.reason.message : "");
      setHistory((h) => [q, ...h.filter((x) => x !== q)].slice(0, 8));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  // the best answer is also results[0]; show it once
  const shown = grounded ? grounded.sources : data?.answer ? [data.answer] : [];
  const others = data ? data.results.filter((r) => !shown.some((s) => s.text === r.text)) : [];

  return (
    <>
      <form onSubmit={(e) => { e.preventDefault(); ask(); }} className="mb-2 flex gap-2">
        <input className="field" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question, e.g. What are the fundamental rights of a citizen?" autoFocus />
        <button className="btn" disabled={busy}>{busy ? "Searching…" : "Search"}</button>
      </form>
      <div className="mb-2 flex flex-wrap gap-4 text-sm text-muted">
        {TYPES.map(([t, label]) => (
          <label key={t}><input type="checkbox" checked={types.includes(t)} onChange={() => toggle(t)} /> {label}</label>
        ))}
      </div>
      {history.length > 0 && (
        <p className="hint">Recent: {history.map((h) => <button key={h} className="linkbtn" onClick={() => ask(h)}>{h}</button>)}</p>
      )}
      {error && <p className="text-bad" role="alert">{error}</p>}

      {data && !data.answer && <p className="hint">No results. Try different words, clear the filters, or ingest some documents first.</p>}
      {busy && <p className="hint" aria-live="polite">Searching and writing an answer…</p>}
      {answerError && <p className="text-bad" role="alert">Could not generate an answer: {answerError}</p>}
      {grounded && (
        <section className="card border-l-4 border-l-accent">
          <h2 className="font-semibold text-accent">Answer</h2>
          <p className="my-2 whitespace-pre-wrap">{grounded.answer}</p>
          {grounded.abstained && <p className="hint">Nothing relevant enough was found, so no answer was generated. Try different words.</p>}
          {grounded.sources.length > 0 && (
            <>
              <h3 className="mt-3 text-xs font-semibold uppercase tracking-wider text-muted">Sources</h3>
              <ol className="mt-1 space-y-1.5 text-sm">
                {grounded.sources.map((s) => (
                  <li key={s.n}><span className="font-semibold text-accent">[{s.n}]</span> {s.citation} <span className="text-muted">· {s.type.replace("_", " ")} · score {s.score.toFixed(3)}</span>
                    <p className="whitespace-pre-wrap text-muted">{s.text}</p></li>
                ))}
              </ol>
            </>
          )}
        </section>
      )}
      {!grounded && data?.answer && (
        <section className="card border-l-4 border-l-accent">
          <h2 className="font-semibold text-accent">Best match</h2>
          <p className="my-2 whitespace-pre-wrap">{data.answer.text}</p>
          <Source citation={data.answer.citation} score={data.answer.score} />
        </section>
      )}
      {others.length > 0 && <h2 className="mb-1 mt-7 font-semibold">{grounded ? "More matches" : "Other matches"}</h2>}
      {others.map((r, i) => (
        <section className="card" key={i}>
          <span className={`badge ${r.type === "qa_pair" ? "!border-accent !bg-soft !text-accent" : r.type === "chunk" ? "!border-ok !text-ok" : ""}`}>{r.type.replace("_", " ")}</span>
          <p className="my-2 whitespace-pre-wrap">{r.text}</p>
          {r.topics?.length > 0 && <p>{r.topics.map((t) => <span className="chip" key={t}>{t}</span>)}</p>}
          <Source citation={r.citation} score={r.score} />
        </section>
      ))}
    </>
  );
}

const Source = ({ citation, score }) => (
  <p className="mt-1.5 text-sm text-muted">Source: {citation} · score {score.toFixed(3)}</p>
);
