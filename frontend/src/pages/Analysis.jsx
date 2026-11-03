import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { ANALYSIS_NOTEBOOK_URL, getAnalysis } from "../api.js";
import { Chart, ChartCard, RETRIEVER_COLOUR, SHORT, Section, Tile } from "../charts.jsx";

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const interval = ([low, high]) => `${f2(low)} to ${f2(high)}`;
const interval3 = ([low, high]) => `${f3(low)} to ${f3(high)}`;  // a small gain needs the third decimal: "-0.00" reads as a typo
const signed = (v) => (v >= 0 ? "+" : "") + f3(v);
const FAMILY = { bm25: "BM25", hybrid: "Hybrid (BM25 + base)", base_rerank: "Base + rerank", base: "Base" };

export default function Analysis() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => { getAnalysis().then(setReport).catch((e) => setError(e.message)); }, []);

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Analysis</h1>
          <p className="hint mt-1">How sure we can be of the evaluation: confidence intervals, tests, a grid search of the settings and cross-validation. The same analysis as the notebook.</p>
        </div>
        <a className="btn inline-flex items-center gap-2" href={ANALYSIS_NOTEBOOK_URL} download><Download size={16} />Download the notebook</a>
      </div>
      {error && <p className="card text-muted" role="alert">{error}</p>}
      {report && <Report r={report} />}
    </div>
  );
}

function Table({ head, rows }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-panel shadow-card">
      <table className="w-full text-sm">
        <thead className="bg-bg text-left text-xs uppercase tracking-wider text-muted"><tr className="[&>th]:px-4 [&>th]:py-3">{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((cells, i) => <tr key={i} className="border-t border-line [&>td]:px-4 [&>td]:py-2.5 [&>td]:tabular-nums">{cells.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

const Verdict = ({ real, yes = "real gap", no = "not shown" }) => (
  <span className={`whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ${real ? "bg-ok/15 text-ok" : "bg-line text-muted"}`}>{real ? yes : no}</span>
);

function Report({ r }) {
  const best = (key) => r.retrievers.reduce((a, b) => (b[key] > a[key] ? b : a));
  return (
    <>
      <div className="mb-8 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile label="Questions" value={r.questions.total} note={`${r.questions.train} train, ${r.questions.test} test`} />
        <Tile label="Searched" value={r.chunks.toLocaleString()} note={`chunks in ${r.sections} sections`} />
        <Tile label={`Best Hit@${r.k} (test)`} value={f2(best("hit").hit)} note={best("hit").label} />
        <Tile label="Best MRR (test)" value={f2(best("mrr").mrr)} note={best("mrr").label} />
      </div>
      <p className="hint -mt-4 mb-8">Computed from the files <code>eval/grid_search.py</code> wrote on {new Date(r.generated).toLocaleString()}. Run it again after changing the index or the questions.</p>

      <Methods r={r} />
      <Comparisons r={r} />
      <Grid grid={r.grid} />
      <CrossValidation cv={r.cross_validation} />
    </>
  );
}

function Intervals({ rows, value, interval: ciKey, title }) {
  const sorted = [...rows].sort((a, b) => a[value] - b[value]);
  return (
    <Chart
      label={`Bar chart of ${title} for each method, with 95% confidence intervals`}
      height={340}
      data={(t) => [{
        type: "bar", orientation: "h", y: sorted.map((x) => x.label), x: sorted.map((x) => x[value]),
        error_x: { type: "data", symmetric: false, array: sorted.map((x) => x[ciKey][1] - x[value]), arrayminus: sorted.map((x) => x[value] - x[ciKey][0]), color: t.muted, thickness: 1.5, width: 4 },
        marker: { color: sorted.map((x) => t[RETRIEVER_COLOUR[x.id]]) }, text: sorted.map((x) => f2(x[value])), textposition: "inside", insidetextanchor: "start", textfont: { color: "#fff" },
        hovertemplate: sorted.map((x) => `${x.full_label}<br>${f2(x[value])} (95% interval ${interval(x[ciKey])})<extra></extra>`),
      }]}
      layout={(t) => ({ showlegend: false, margin: { l: 120, r: 16, t: 8, b: 48 }, xaxis: { range: [0, 1.05], title: { text: title, font: { color: t.muted } } }, yaxis: { automargin: true } })}
    />
  );
}

function Methods({ r }) {
  return (
    <Section title="Search methods on the test questions" hint="Test questions are from Parts of the document that no method was tuned on. Bars show the 95% confidence interval: with few questions it is wide, and bars that overlap cannot be told apart.">
      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <ChartCard title={`Questions answered in the top ${r.k} (Hit@${r.k})`}><Intervals rows={r.retrievers} value="hit" interval="hit_ci" title={`Hit@${r.k}`} /></ChartCard>
        <ChartCard title="Mean reciprocal rank (MRR)"><Intervals rows={r.retrievers} value="mrr" interval="mrr_ci" title="MRR" /></ChartCard>
      </div>
      <Table
        head={["Method", `Recall@${r.k}`, "MRR", `nDCG@${r.k}`, `Hit@${r.k} (95% interval)`, "Right section first"]}
        rows={r.retrievers.map((x) => [x.full_label, f3(x.recall), f3(x.mrr), f3(x.ndcg), `${f2(x.hit)} (${interval(x.hit_ci)})`, `${Math.round(x.first * 100)}%`])}
      />
      <p className="hint mt-2">Recall@{r.k} counts the share of right sections found; Hit@{r.k} counts a question as answered if any right section is in the top {r.k}, which is what the intervals are drawn for. MRR is 1 divided by the position of the first right section, averaged.</p>
    </Section>
  );
}

function Comparisons({ r }) {
  const rows = (list) => list.map((c) => [`${SHORT[c.first]} vs ${SHORT[c.second]}`, c.questions, signed(c.difference), interval(c.ci), c.p_adjusted < 0.001 ? "< 0.001" : f3(c.p_adjusted), <Verdict key="v" real={c.real} />]);
  const head = ["Comparison", "Questions", "Difference in MRR", "95% interval", "p (adjusted)"];
  return (
    <Section title="Are the differences real?" hint="Each method is compared with another on the same questions. A gap is called real when its 95% interval excludes 0. The p-value is from a Wilcoxon signed-rank test, multiplied by the number of comparisons (Bonferroni).">
      <h3 className="mb-2 text-sm font-medium">Test questions (all methods)</h3>
      <Table head={[...head, "Verdict"]} rows={rows(r.comparisons.test)} />
      <h3 className="mb-2 mt-5 text-sm font-medium">All questions (methods that were not trained)</h3>
      <Table head={[...head, "Verdict"]} rows={rows(r.comparisons.all)} />
      <p className="hint mt-2">“Not shown” does not mean “no difference”: it means these questions are too few to tell. The fine-tuned model is compared on the test questions only, because it was trained on the others.</p>
    </Section>
  );
}

function Grid({ grid }) {
  const family = (name) => grid.filter((g) => g.family === name);
  const bm25 = family("bm25");
  const k1s = [...new Set(bm25.map((g) => g.params.k1))].sort((a, b) => a - b);
  const bs = [...new Set(bm25.map((g) => g.params.b))].sort((a, b) => a - b);
  const cell = (k1, b) => bm25.find((g) => g.params.k1 === k1 && g.params.b === b);
  const lo = Math.min(...bm25.map((g) => g.mrr));
  const hi = Math.max(...bm25.map((g) => g.mrr));
  const NAMES = { recall: "Recall@5", mrr: "MRR", ndcg: "nDCG@5" };

  const lines = (rows, key, label) => (
    <Chart
      label={`Line chart of Recall, MRR and nDCG against ${label}`}
      height={300}
      data={(t) => ["recall", "mrr", "ndcg"].map((m, i) => ({
        type: "scatter", mode: "lines+markers", name: NAMES[m], x: rows.map((g) => g.params[key]), y: rows.map((g) => g[m]), line: { color: t[`s${i + 1}`] },
        hovertemplate: `${label} %{x}: %{y:.3f}<extra>${NAMES[m]}</extra>`,
      }))}
      layout={(t) => ({ xaxis: { title: { text: label, font: { color: t.muted } } }, yaxis: { range: [0.3, 1], title: { text: "Mean score", font: { color: t.muted } } }, margin: { l: 56, r: 16, t: 8, b: 100 }, legend: { orientation: "h", y: -0.5, font: { color: t.muted } } })}
    />
  );

  return (
    <Section title="Grid search" hint="Every setting of three methods was scored on every question. The settings are not learned from data, so they are searched: BM25's k1 and b, the hybrid's fusion constant, and how many top results the reranker re-scores.">
      <div className="grid gap-4 lg:grid-cols-3">
        <ChartCard title="BM25: MRR for each k1 and b">
          <table className="mt-2 w-full text-sm tabular-nums">
            <thead><tr><th className="p-1 text-left text-xs font-normal text-muted">k1 \ b</th>{bs.map((b) => <th key={b} className="p-1 text-xs font-normal text-muted">{b}</th>)}</tr></thead>
            <tbody>
              {k1s.map((k1) => (
                <tr key={k1}>
                  <th className="p-1 text-left text-xs font-normal text-muted">{k1}</th>
                  {bs.map((b) => { const g = cell(k1, b); const share = hi > lo ? (g.mrr - lo) / (hi - lo) : 0; return <td key={b} className="p-2 text-center" style={{ background: `color-mix(in srgb, var(--s1) ${Math.round(12 + share * 55)}%, transparent)` }}>{f3(g.mrr)}</td>; })}
                </tr>
              ))}
            </tbody>
          </table>
        </ChartCard>
        <ChartCard title="Hybrid: fusion constant">{lines(family("hybrid").sort((a, b) => a.params.rrf_k - b.params.rrf_k), "rrf_k", "rrf_k")}</ChartCard>
        <ChartCard title="Base + rerank: results re-scored">{lines(family("base_rerank").sort((a, b) => a.params.depth - b.params.depth), "depth", "depth")}</ChartCard>
      </div>
    </Section>
  );
}

function CrossValidation({ cv }) {
  const count = (list) => Object.entries(list.reduce((n, c) => ({ ...n, [c]: (n[c] ?? 0) + 1 }), {})).map(([c, n]) => `${c} (${n} of ${list.length})`).join(", ");
  const names = cv.families.map((f) => f.family);
  return (
    <Section title="Cross-validation by Part" hint={`The questions are split into ${cv.folds} folds so that all questions of a Part stay together. The best setting is chosen on the other folds and scored on the held-out one, so no setting is judged on a Part it was tuned on. Every question is held out once.`}>
      <Table
        head={["Method", "Default setting", "MRR, default", "MRR, tuned in cross-validation", "Gain (95% interval)", "Verdict", "Setting chosen in each fold"]}
        rows={cv.families.map((f) => [FAMILY[f.family], f.default_setting, f3(f.default.mrr), f3(f.tuned.mrr), f.family === "base" ? "none to tune" : `${signed(f.gain.mrr)} (${interval3(f.gain.ci)})`,
          f.family === "base" ? "" : <Verdict key="v" real={f.gain.real} yes="real gain" />, f.family === "base" ? "as downloaded" : count(f.chosen)])}
      />
      <h3 className="mb-2 mt-5 text-sm font-medium">MRR in each held-out fold</h3>
      <Table
        head={["Fold", "Parts held out", ...names.map((n) => FAMILY[n])]}
        rows={cv.held_out_parts.map((parts, i) => [i + 1, parts.join(", "), ...cv.families.map((f) => f3(f.fold_mrr[i]))])}
      />
      <p className="hint mt-2">Compare a gain with the spread between folds: a gain smaller than that spread is easily luck, because which Parts are held out can matter more than the settings. The fine-tuned model is not part of this section: it was trained on the training Parts.</p>
    </Section>
  );
}
