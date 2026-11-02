import { Fragment, useEffect, useState } from "react";
import { Download, Loader2, Play } from "lucide-react";
import { getEvalQuestions, getEvalResults, getStats, runEval, watchEval } from "../api.js";
import { Chart, RETRIEVER_COLOUR } from "../charts.jsx";

const TYPE_LABELS = { chunk: "Chunks", qa_pair: "Q&A pairs", factoid: "Factoids", triplet: "Triplets", topic: "Glossary", summary: "Summaries" };
const pct = (v) => v.toFixed(3);

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [results, setResults] = useState(null);
  const [retrievers, setRetrievers] = useState([]); // what the server can run: [{id, label}]
  const [error, setError] = useState("");

  useEffect(() => {
    getStats().then(setStats).catch((e) => setError(e.message));
    getEvalResults().then(setResults).catch((e) => setError(e.message));
    getEvalQuestions().then((q) => setRetrievers(q.retrievers)).catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
      <p className="hint mb-6 mt-1">How well retrieval finds the right passage, and what is in the index.</p>
      {error && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      <Evaluation results={results} retrievers={retrievers} onDone={setResults} />
      <IndexStats stats={stats} />
    </div>
  );
}

function Section({ title, hint, children }) {
  return (
    <section className="mb-10">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {hint && <p className="hint mb-4">{hint}</p>}
      {children}
    </section>
  );
}

function Tile({ label, value, note }) {
  return (
    <div className="card my-0 min-w-0">
      <p className="hint truncate">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{value}</p>
      {note && <p className="hint truncate">{note}</p>}
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="card my-0 min-w-0">
      <h3 className="mb-1 text-sm font-medium">{title}</h3>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------- evaluation

function Evaluation({ results, retrievers, onDone }) {
  const [chosen, setChosen] = useState(null); // null = all available
  const [k, setK] = useState(5);
  const [split, setSplit] = useState("test");
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const running = job?.status === "running";
  const picked = chosen ?? retrievers.map((r) => r.id);

  async function run() {
    setError("");
    try {
      const { job_id } = await runEval(picked, k, split);
      const done = await watchEval(job_id, setJob);
      if (done.status === "failed") throw new Error(done.error);
      onDone(await getEvalResults());
    } catch (e) {
      setError(e.message);
      setJob(null);
    }
  }
  const toggle = (id) => setChosen(picked.includes(id) ? picked.filter((p) => p !== id) : [...picked, id]);
  const has = results && results.retrievers;

  return (
    <Section title="Retrieval evaluation" hint="Each question is answered by the section that holds its evidence text. Recall@k: share of right sections in the top k. MRR: 1 / rank of the first right section. nDCG@k: rewards ranking right sections higher.">
      <div className="card mb-4 mt-0 flex flex-wrap items-end gap-x-6 gap-y-3">
        <fieldset className="flex flex-wrap gap-x-4 gap-y-1">
          <legend className="hint mb-1">Retrievers</legend>
          {retrievers.map((r) => (
            <label key={r.id} className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)} disabled={running} />
              <i className="size-2.5 rounded-sm" style={{ background: `var(--${RETRIEVER_COLOUR[r.id]})` }} />{r.label}
            </label>
          ))}
        </fieldset>
        <label className="text-sm"><span className="hint mb-1 block">k</span>
          <select className="field !w-20 !py-1.5" value={k} onChange={(e) => setK(Number(e.target.value))} disabled={running}>
            {[1, 3, 5, 10, 20].map((n) => <option key={n}>{n}</option>)}
          </select>
        </label>
        <label className="text-sm"><span className="hint mb-1 block">Questions</span>
          <select className="field !w-auto !py-1.5" value={split} onChange={(e) => setSplit(e.target.value)} disabled={running}>
            <option value="test">Test split (held-out Parts)</option><option value="train">Train split</option><option value="all">All</option>
          </select>
        </label>
        <button className="btn inline-flex items-center gap-2" disabled={running || !picked.length} onClick={run}>
          {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}{running ? "Running" : "Run evaluation"}
        </button>
        {running && <span className="hint" role="status">{job.message} ({job.done}/{job.total})</span>}
      </div>
      {error && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      {has ? <Results r={results} /> : <p className="card text-muted">No evaluation has run yet. Pick retrievers and press Run evaluation.</p>}
    </Section>
  );
}

function Results({ r }) {
  const ids = Object.keys(r.retrievers);
  const best = (metric) => ids.reduce((a, b) => (r.retrievers[b][metric] > r.retrievers[a][metric] ? b : a));
  const tiles = [["recall", `Recall@${r.k}`], ["mrr", "MRR"], ["ndcg", `nDCG@${r.k}`]];
  const metricNames = tiles.map(([, name]) => name);

  return (
    <>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile label="Questions scored" value={r.questions} note={r.unresolved.length ? `${r.unresolved.length} skipped: evidence not found` : `${r.sections} sections, ${r.chunks} chunks`} />
        {tiles.map(([m, name]) => <Tile key={m} label={`Best ${name}`} value={pct(r.retrievers[best(m)][m])} note={r.retrievers[best(m)].label} />)}
      </div>
      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <ChartCard title="Retrieval quality by retriever">
          <Chart
            label="Grouped bar chart of Recall, MRR and nDCG for each retriever"
            data={(t) => ids.map((id) => ({
              type: "bar", name: r.retrievers[id].label, x: metricNames, y: tiles.map(([m]) => r.retrievers[id][m]),
              text: tiles.map(([m]) => pct(r.retrievers[id][m])), textposition: "outside", textfont: { color: t.text }, cliponaxis: false,
              marker: { color: t[RETRIEVER_COLOUR[id]], line: { width: 2, color: t.panel } }, hovertemplate: "%{x}: %{y:.3f}<extra>" + r.retrievers[id].label + "</extra>",
            }))}
            layout={() => ({ barmode: "group", bargap: 0.4, bargroupgap: 0.06, yaxis: { range: [0, 1.12], tickformat: ".1f" }, xaxis: {} })}
          />
        </ChartCard>
        <ChartCard title="Recall as k grows">
          <Chart
            label="Line chart of Recall at k, for k from 1 to 20, per retriever"
            data={(t) => ids.map((id) => ({
              type: "scatter", mode: "lines+markers", name: r.retrievers[id].label, x: r.retrievers[id].curve.map((_, i) => i + 1), y: r.retrievers[id].curve,
              line: { color: t[RETRIEVER_COLOUR[id]], width: 2 }, marker: { size: 8, color: t[RETRIEVER_COLOUR[id]], line: { width: 2, color: t.panel } },
              hovertemplate: "Recall@%{x}: %{y:.3f}<extra>" + r.retrievers[id].label + "</extra>",
            }))}
            layout={(t) => ({
              hovermode: "x unified", margin: { l: 56, r: 16, t: 24, b: 72 }, legend: { orientation: "h", y: -0.32, font: { color: t.muted } }, yaxis: { range: [0, 1.05], tickformat: ".1f", title: { text: "Recall" } }, xaxis: { title: { text: "k (top results considered)" }, dtick: 2 },
              shapes: [{ type: "line", x0: r.k, x1: r.k, y0: 0, y1: 1, line: { color: t.muted, width: 1, dash: "dot" } }],
            })}
          />
        </ChartCard>
      </div>
      <Questions r={r} ids={ids} />
    </>
  );
}

function csv(r, ids) {
  const cell = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const rows = [["id", "question", "split", ...ids.map((i) => `rank_${i}`)]];
  r.per_question.forEach((q) => rows.push([q.id, q.question, q.split, ...ids.map((i) => q.ranks[i] ?? "")]));
  rows.push([], ["retriever", `recall@${r.k}`, "mrr", `ndcg@${r.k}`]);
  ids.forEach((i) => rows.push([r.retrievers[i].label, ...["recall", "mrr", "ndcg"].map((m) => r.retrievers[i][m].toFixed(4))]));
  return rows.map((row) => row.map(cell).join(",")).join("\n");
}

function RankCell({ rank, k }) {
  if (rank == null) return <span className="rounded-md bg-bad/10 px-2 py-0.5 text-bad">miss</span>;
  const tone = rank === 1 ? "bg-ok/15 text-ok" : rank <= k ? "bg-soft text-accent" : "bg-bad/10 text-bad";
  return <span className={`rounded-md px-2 py-0.5 tabular-nums ${tone}`}>{rank === 1 ? "1st" : `#${rank}`}{rank > k ? " (beyond k)" : ""}</span>;
}

function Questions({ r, ids }) {
  const [onlyMisses, setOnlyMisses] = useState(false);
  const [open, setOpen] = useState(null);
  const rows = r.per_question.filter((q) => !onlyMisses || ids.some((i) => q.ranks[i] == null || q.ranks[i] > r.k));
  const download = () => {
    const url = URL.createObjectURL(new Blob([csv(r, ids)], { type: "text/csv" }));
    Object.assign(document.createElement("a"), { href: url, download: "evaluation.csv" }).click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card my-0">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <h3 className="text-sm font-medium">Per question: where each retriever ranks the right section</h3>
        <label className="ml-auto flex items-center gap-2 text-sm"><input type="checkbox" checked={onlyMisses} onChange={(e) => setOnlyMisses(e.target.checked)} />Only misses</label>
        <button className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-bg" onClick={download}><Download size={14} />CSV</button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="hint"><tr>
            <th className="py-2 pr-3 font-medium">Question</th>
            {ids.map((i) => <th key={i} className="whitespace-nowrap px-3 py-2 font-medium"><i className="mr-1.5 inline-block size-2.5 rounded-sm" style={{ background: `var(--${RETRIEVER_COLOUR[i]})` }} />{r.retrievers[i].label}</th>)}
          </tr></thead>
          <tbody>
            {rows.map((q) => (
              <Fragment key={q.id}>
                <tr className="cursor-pointer border-t border-line hover:bg-bg" onClick={() => setOpen(open === q.id ? null : q.id)} aria-expanded={open === q.id}>
                  <td className="py-2 pr-3">{q.question}<span className="hint ml-2">{q.id}</span></td>
                  {ids.map((i) => <td key={i} className="whitespace-nowrap px-3 py-2"><RankCell rank={q.ranks[i]} k={r.k} /></td>)}
                </tr>
                {open === q.id && (
                  <tr className="bg-bg"><td colSpan={ids.length + 1} className="px-3 py-3">
                    <p className="hint mb-2">Right section: {q.relevant.join(" | ")}</p>
                    <div className="grid gap-3 md:grid-cols-3">
                      {ids.map((i) => (
                        <div key={i}><p className="mb-1 text-xs font-medium">{r.retrievers[i].label}: top {q.top[i].length}</p>
                          <ol className="space-y-0.5 text-xs">
                            {q.top[i].map((t, n) => <li key={n} className={t.hit ? "font-medium text-ok" : "text-muted"}>{n + 1}. {t.section || "(no heading)"}{t.page ? `, p. ${t.page}` : ""}{t.hit ? " ✓ right section" : ""}</li>)}
                          </ol>
                        </div>
                      ))}
                    </div>
                  </td></tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        {!rows.length && <p className="py-4 text-center text-muted">No misses: every retriever ranked the right section within k.</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- index

function IndexStats({ stats }) {
  if (!stats) return <Section title="Index"><p className="hint">Loading…</p></Section>;
  const types = Object.entries(stats.points).filter(([, n]) => n > 0);
  const parts = stats.chunks_per_part.slice(0, 12).reverse();
  return (
    <Section title="Index" hint="What the knowledge base holds: the pieces made from your documents.">
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Tile label="Documents" value={stats.documents} />
        <Tile label="Sections (parents)" value={stats.parents} />
        {["chunk", "qa_pair", "factoid"].map((t) => <Tile key={t} label={TYPE_LABELS[t]} value={(stats.points[t] ?? 0).toLocaleString()} />)}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Points by type">
          <Chart
            label="Bar chart of stored points by type"
            data={(t) => [{ type: "bar", orientation: "h", y: types.map(([k]) => TYPE_LABELS[k]), x: types.map(([, n]) => n), text: types.map(([, n]) => n.toLocaleString()), textposition: "outside", cliponaxis: false,
              marker: { color: t.s1, line: { width: 2, color: t.panel } }, hovertemplate: "%{y}: %{x:,}<extra></extra>" }]}
            layout={() => ({ margin: { l: 90, r: 56, t: 8, b: 40 }, yaxis: { autorange: "reversed" }, xaxis: { tickformat: "," }, showlegend: false })}
            height={280}
          />
        </ChartCard>
        <ChartCard title="Child chunk size (words)">
          <Chart
            label="Histogram of child chunk sizes in words"
            data={(t) => [{ type: "bar", x: stats.child_words.map((b) => b.from + 5), width: 9, y: stats.child_words.map((b) => b.count), customdata: stats.child_words.map((b) => `${b.from}–${b.to}`),
              marker: { color: t.s1, line: { width: 2, color: t.panel } }, hovertemplate: "%{customdata} words: %{y} chunks<extra></extra>" }]}
            layout={() => ({ margin: { l: 56, r: 16, t: 8, b: 56 }, xaxis: { title: { text: "Words per child chunk" }, dtick: 20, tickangle: 0 }, yaxis: { title: { text: "Chunks" } }, showlegend: false })}
            height={280}
          />
        </ChartCard>
        <div className="lg:col-span-2">
          <ChartCard title="Chunks per Part of the document (largest 12)">
            <Chart
              label="Bar chart of chunks per Part"
              data={(t) => [{ type: "bar", orientation: "h", y: parts.map((p) => p.part), x: parts.map((p) => p.chunks), text: parts.map((p) => p.chunks), textposition: "outside", cliponaxis: false,
                marker: { color: t.s1, line: { width: 2, color: t.panel } }, hovertemplate: "%{y}: %{x} chunks<extra></extra>" }]}
              layout={() => ({ margin: { l: 280, r: 40, t: 8, b: 40 }, showlegend: false, yaxis: { automargin: true } })}
              height={400}
            />
          </ChartCard>
        </div>
      </div>
    </Section>
  );
}
