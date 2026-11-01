import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ask as askApi, getSuggestions, search } from "../api.js";

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
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => { getSuggestions().then((d) => setSuggestions(d.suggestions)).catch(() => {}); }, []);

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
  // With a grounded answer the matches are the LLM's readable rewrites; if that failed, fall back to the raw results
  const shown = data?.answer ? [data.answer] : [];
  const others = grounded ? grounded.matches ?? [] : data ? data.results.filter((r) => !shown.some((s) => s.text === r.text)) : [];

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
      {suggestions.length > 0 && (
        <details className={data ? "mb-3" : "my-4"} open={!data}>
          <summary className="cursor-pointer text-sm font-medium text-muted">Suggested questions</summary>
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {suggestions.map((s) => (
              <button key={s.question} type="button" disabled={busy} onClick={() => ask(s.question)}
                className="rounded-xl border border-line bg-panel p-3 text-left shadow-sm transition-colors hover:border-accent hover:bg-soft disabled:opacity-50">
                <span className="block text-xs font-semibold uppercase tracking-wider text-accent">{s.topic}</span>
                <span className="block text-sm text-ink">{s.question}</span>
              </button>
            ))}
          </div>
        </details>
      )}
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
          <AnswerText text={grounded.answer} />
          {grounded.abstained && <p className="hint">Nothing relevant enough was found, so no answer was generated. Try different words.</p>}
          {grounded.sources.length > 0 && (
            <p className="mt-3 text-sm text-muted">
              Sources: {grounded.sources.map((src) => (
                <a key={src.n} href={`#match-${src.n}`} className="mr-2 text-accent underline hover:no-underline">[{src.n}] {where(src)}</a>
              ))}
            </p>
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
      {others.length > 0 && <h2 className="mb-1 mt-7 font-semibold">{grounded ? `More matches (top ${others.length})` : "Other matches"}</h2>}
      {others.map((r, i) => (
        <section className="card scroll-mt-20" key={r.n ?? i} id={r.n ? `match-${r.n}` : undefined}>
          <div className="flex items-center gap-2">
            {r.n && <span className="font-semibold text-accent">[{r.n}]</span>}
            <span className={`badge ${r.type === "qa_pair" ? "!border-accent !bg-soft !text-accent" : r.type === "chunk" ? "!border-ok !text-ok" : ""}`}>{r.type.replace("_", " ")}</span>
            <span className="text-sm text-muted">{where(r)} · score {r.score.toFixed(3)}</span>
            {r.cited && <span className="ml-auto text-xs text-ok">used in the answer</span>}
          </div>
          {r.answer ? <AnswerText text={r.answer} /> : <MatchText item={r} />}
          {r.topics?.length > 0 && <p className="mt-2">{r.topics.map((t) => <span className="chip" key={t}>{t}</span>)}</p>}
          <PassageLink item={r} />
        </section>
      ))}
      {grounded && others.length === 0 && grounded.sources.length > 0 && <p className="hint mt-4">No further passages helped answer this.</p>}
    </>
  );
}

const Source = ({ citation, score }) => (
  <p className="mt-1.5 text-sm text-muted">Source: {citation} · score {score.toFixed(3)}</p>
);

// A short preview: Q&A pairs as question and answer, long passages cut to a few lines (the link shows the rest)
function MatchText({ item }) {
  if (item.type === "qa_pair") {
    return <p className="mt-2"><strong>{item.question}</strong><br />{item.answer}</p>;
  }
  return <p className="mt-2 line-clamp-4 whitespace-pre-line">{item.text}</p>;
}

// Opens the passage the result came from (for a Q&A pair, the passage it was written from)
function PassageLink({ item }) {
  if (!item.document_id || item.parent_id == null) return null;
  return <Link className="mt-2 inline-block text-sm text-accent underline hover:no-underline" to={`/passage/${item.document_id}/${item.parent_id}`}>View passage →</Link>;
}

// paragraphs and "- " bullet lists, which is how the answer prompt asks the LLM to write
function AnswerText({ text }) {
  const blocks = [];
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)/);
    const last = blocks[blocks.length - 1];
    if (bullet) {
      if (last?.list) last.items.push(bullet[1]);
      else blocks.push({ list: true, items: [bullet[1]] });
    } else if (line.trim()) blocks.push({ text: line });
  }
  return (
    <div className="my-2 space-y-2">
      {blocks.map((b, i) => b.list
        ? <ul key={i} className="list-disc space-y-1 pl-5">{b.items.map((t, j) => <li key={j}>{inline(t)}</li>)}</ul>
        : <p key={i}>{inline(b.text)}</p>)}
    </div>
  );
}

// **bold** only; the LLM sometimes uses it for the heading of a bullet
const inline = (t) => t.split(/\*\*(.+?)\*\*/).map((part, i) => (i % 2 ? <strong key={i}>{part}</strong> : part));

// "Constitution of India (2024).pdf · Part III › Right to Freedom · p. 41"
const where = (r) => [r.citation, r.section, r.page ? `p. ${r.page}` : ""].filter(Boolean).join(" · ");
