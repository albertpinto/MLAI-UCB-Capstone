import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRight, Check, Clock, Copy, Info, Loader2, Search, Sparkles } from "lucide-react";
import { ask as askApi, getSuggestions, search } from "../api.js";

const TYPES = [
  ["qa_pair", "Q&A"],
  ["chunk", "Passages"],
  ["summary", "Summaries"],
  ["topic", "Topics"],
];
const STEPS = ["Searching the documents", "Reading the matching sections", "Writing the answer"];

export default function Ask() {
  const [params, setParams] = useSearchParams();
  const urlQuery = params.get("q") ?? "";
  const [query, setQuery] = useState(urlQuery);
  const [types, setTypes] = useState([]);
  const [result, setResult] = useState(null); // { q, data, grounded, answerError }
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  const [active, setActive] = useState(null); // the source card a citation just pointed at
  const ran = useRef("");

  useEffect(() => { getSuggestions().then((d) => setSuggestions(d.suggestions)).catch(() => {}); }, []);

  // the question lives in the URL (?q=...), so it can be bookmarked, shared and reloaded
  useEffect(() => {
    if (urlQuery && urlQuery !== ran.current) { ran.current = urlQuery; setQuery(urlQuery); run(urlQuery); }
  }, [urlQuery]);

  // purely visual progress while the request runs
  useEffect(() => {
    if (!busy) return;
    setStep(0);
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 1800);
    return () => clearInterval(timer);
  }, [busy]);

  const toggle = (t) => setTypes((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));
  const submit = (q) => {
    if (!q.trim()) return;
    if (q === urlQuery) { ran.current = q; run(q); } else setParams({ q });
  };

  async function run(q) {
    setBusy(true);
    setError("");
    try {
      // the grounded answer and the plain results are independent: results still show if the LLM call fails
      const [found, answer] = await Promise.allSettled([search(q, types), askApi(q, types)]);
      if (found.status === "rejected") throw found.reason;
      setResult({ q, data: found.value, grounded: answer.status === "fulfilled" ? answer.value : null, answerError: answer.status === "rejected" ? answer.reason.message : "" });
      setHistory((h) => [q, ...h.filter((x) => x !== q)].slice(0, 6));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function goToSource(n) {
    setActive(n);
    document.getElementById(`match-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => setActive(null), 1600);
  }

  const { data, grounded, answerError } = result ?? {};
  // with a grounded answer the sources are the LLM's readable rewrites; if that failed, fall back to the raw results
  const sources = grounded ? grounded.matches ?? [] : data?.results ?? [];
  const showHero = !result && !busy;

  return (
    <div>
      <div className={`mx-auto max-w-3xl text-center ${showHero ? "pt-6 md:pt-12" : ""}`}>
        {showHero && (
          <>
            <span className="mx-auto mb-4 grid size-12 place-items-center rounded-2xl bg-linear-to-br from-indigo-600 to-violet-600 text-white shadow-card"><Sparkles size={22} /></span>
            <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">What would you like to know?</h1>
            <p className="mx-auto mt-2 max-w-xl text-muted">Ask in everyday language. Answers come only from your documents, with citations you can open.</p>
          </>
        )}
        <form onSubmit={(e) => { e.preventDefault(); submit(query); }} className={`flex gap-2 text-left ${showHero ? "mt-7" : ""}`}>
          <div className="relative flex-1">
            <Search size={18} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
            <input className="field !rounded-2xl !py-3.5 !pl-11 text-base" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. What are the fundamental rights of a citizen?" autoFocus aria-label="Your question" />
          </div>
          <button className="btn !rounded-2xl !px-6" disabled={busy}>{busy ? "Asking…" : "Ask"}</button>
        </form>
        <div className="mt-3 flex flex-wrap items-center justify-center gap-2 text-sm" role="group" aria-label="Search in">
          <span className="text-muted">Search in</span>
          <Chip active={types.length === 0} onClick={() => setTypes([])}>Everything</Chip>
          {TYPES.map(([t, label]) => <Chip key={t} active={types.includes(t)} onClick={() => toggle(t)}>{label}</Chip>)}
        </div>
        {history.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center justify-center gap-2 text-sm">
            <Clock size={14} className="text-muted" />
            {history.map((h) => <button key={h} onClick={() => submit(h)} className="max-w-[16rem] truncate rounded-full border border-line bg-panel px-3 py-1 text-muted transition hover:border-accent hover:text-accent">{h}</button>)}
          </div>
        )}
      </div>

      {error && <p className="mx-auto mt-6 max-w-3xl rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}

      {busy && (
        <ol className="mx-auto mt-8 flex max-w-3xl flex-wrap justify-center gap-x-6 gap-y-2 text-sm" aria-live="polite">
          {STEPS.map((label, i) => (
            <li key={label} className={`flex items-center gap-2 ${i <= step ? "text-ink" : "text-muted/60"}`}>
              {i < step ? <Check size={16} className="text-ok" /> : i === step ? <Loader2 size={16} className="animate-spin text-accent" /> : <span className="size-4 rounded-full border border-line" />}
              {label}
            </li>
          ))}
        </ol>
      )}

      {showHero && suggestions.length > 0 && (
        <section className="mx-auto mt-10 max-w-4xl">
          <h2 className="mb-3 text-center text-sm font-medium uppercase tracking-wider text-muted">Try asking</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {suggestions.map((s) => (
              <button key={s.question} type="button" onClick={() => submit(s.question)}
                className="group flex items-start justify-between gap-3 rounded-2xl border border-line bg-panel p-4 text-left shadow-card transition hover:-translate-y-0.5 hover:border-accent">
                <span>
                  <span className="block text-xs font-semibold uppercase tracking-wider text-accent">{s.topic}</span>
                  <span className="mt-1 block text-[15px] leading-snug">{s.question}</span>
                </span>
                <ArrowRight size={18} className="mt-1 shrink-0 text-muted transition group-hover:translate-x-0.5 group-hover:text-accent" />
              </button>
            ))}
          </div>
        </section>
      )}

      {result && !busy && (
        <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <div>
            {answerError && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">Could not write an answer ({answerError}). The matching passages are shown instead.</p>}
            {grounded && <AnswerCard grounded={grounded} onCite={goToSource} />}
            {!grounded && !data.results.length && <Empty>Nothing matched. Try different words or clear the filters.</Empty>}
          </div>
          <aside className="lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] lg:self-start lg:overflow-y-auto lg:pr-1">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold">Sources <span className="rounded-full bg-soft px-2 py-0.5 text-xs font-medium text-accent">{sources.length}</span></h2>
            {sources.length === 0 && <p className="hint">{grounded ? "No passages helped answer this." : "No results."}</p>}
            {sources.map((s, i) => <SourceCard key={s.n ?? i} source={s} highlighted={active === s.n} />)}
          </aside>
        </div>
      )}
    </div>
  );
}

const Chip = ({ active, ...props }) => (
  <button type="button" aria-pressed={active} className={`rounded-full border px-3 py-1 transition ${active ? "border-accent bg-soft font-medium text-accent" : "border-line bg-panel text-muted hover:text-ink"}`} {...props} />
);

const Empty = ({ children }) => (
  <div className="card flex items-center gap-3 text-muted"><Info size={18} />{children}</div>
);

function AnswerCard({ grounded, onCite }) {
  const [copied, setCopied] = useState(false);
  const copy = () => navigator.clipboard?.writeText(grounded.answer.replace(/\s*\[\d+\]/g, "")).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  if (grounded.abstained) {
    return <Empty>{grounded.answer} Try different words, or check that the document covers this.</Empty>;
  }
  return (
    <article className="card !my-0 border-t-4 border-t-indigo-600">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="flex items-center gap-2 font-semibold text-accent"><Sparkles size={18} />Answer</h2>
        <button onClick={copy} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-muted transition hover:bg-soft hover:text-accent" aria-label="Copy answer">
          {copied ? <><Check size={14} />Copied</> : <><Copy size={14} />Copy</>}
        </button>
      </header>
      <AnswerText text={grounded.answer} onCite={onCite} />
      <p className="mt-4 border-t border-line pt-3 text-xs text-muted">Based on {grounded.sources.length} passage{grounded.sources.length === 1 ? "" : "s"} from your documents. Select a number to see its source.</p>
    </article>
  );
}

function SourceCard({ source: r, highlighted }) {
  return (
    <section id={r.n ? `match-${r.n}` : undefined} className={`card !my-0 mb-3 scroll-mt-28 !p-4 transition ${highlighted ? "ring-2 ring-accent" : ""}`}>
      <div className="flex items-center gap-2 text-xs">
        {r.n && <span className="grid size-5 shrink-0 place-items-center rounded-md bg-soft font-semibold text-accent">{r.n}</span>}
        <span className="truncate font-medium text-ink">{r.citation}</span>
        {r.cited && <span className="ml-auto inline-flex shrink-0 items-center gap-1 text-ok"><Check size={13} />used</span>}
      </div>
      {(r.section || r.page) && <p className="mt-1 truncate text-xs text-muted">{[r.section, r.page ? `page ${r.page}` : ""].filter(Boolean).join(" · ")}</p>}
      <div className="mt-2 text-sm">{r.n && r.answer ? <AnswerText text={r.answer} /> : <RawText item={r} />}</div>
      <div className="mt-3 flex items-center gap-3">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-line" title={`similarity ${r.score.toFixed(3)}`}><div className="h-full rounded-full bg-linear-to-r from-indigo-600 to-violet-600" style={{ width: `${Math.round(Math.min(Math.max(r.score, 0), 1) * 100)}%` }} /></div>
        {r.document_id && r.parent_id != null && (
          <Link className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-accent hover:underline" to={`/passage/${r.document_id}/${r.parent_id}`}>View passage <ArrowRight size={13} /></Link>
        )}
      </div>
    </section>
  );
}

// the raw fallback when no readable rewrite exists: Q&A pairs as question and answer, long passages cut short
function RawText({ item }) {
  if (!item.n && item.type === "qa_pair") return <p><strong>{item.question}</strong><br />{item.answer}</p>;  // a raw search result, not a card the LLM rewrote
  return <p className="line-clamp-4 whitespace-pre-line">{item.text}</p>;
}

// paragraphs and "- " bullet lists, which is how the answer prompt asks the LLM to write
function AnswerText({ text, onCite }) {
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
    <div className="space-y-2.5 leading-relaxed">
      {blocks.map((b, i) => b.list
        ? <ul key={i} className="list-disc space-y-1.5 pl-5 marker:text-accent">{b.items.map((t, j) => <li key={j}>{inline(t, onCite)}</li>)}</ul>
        : <p key={i}>{inline(b.text, onCite)}</p>)}
    </div>
  );
}

// **bold**, and [1][3] citations as small numbered chips that point at the source card
function inline(text, onCite) {
  return text.split(/(\*\*.+?\*\*|\[\d+\])/).map((part, i) => {
    if (part.startsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    const cite = part.match(/^\[(\d+)\]$/);
    if (!cite) return part;
    const n = Number(cite[1]);
    return onCite
      ? <button key={i} onClick={() => onCite(n)} className="ml-0.5 mr-px inline-grid size-5 -translate-y-px place-items-center rounded-md bg-soft align-middle text-[11px] font-semibold text-accent transition hover:bg-accent hover:text-white" aria-label={`Source ${n}`}>{n}</button>
      : <span key={i} className="text-muted">[{n}]</span>;
  });
}
