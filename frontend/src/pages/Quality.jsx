import { useEffect, useState } from "react";
import { Check, X } from "lucide-react";
import { getQuality, reviewArtifact, runAnswerEval, runArtifactCheck, watchQualityJob } from "../api.js";

const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
const KIND = { qa_pair: "Q&A pairs", factoid: "Factoids" };

function Tile({ label, value, hint }) {
  return (
    <div className="card !my-0 text-center">
      <p className="text-2xl font-semibold tabular-nums">{value}</p>
      <p className="text-sm">{label}</p>
      {hint && <p className="hint">{hint}</p>}
    </div>
  );
}

export default function Quality() {
  const [data, setData] = useState({});
  const [busy, setBusy] = useState("");
  const [sample, setSample] = useState(50);
  const [kind, setKind] = useState("qa_pair");
  const [show, setShow] = useState("unreviewed");
  const [error, setError] = useState("");

  const refresh = () => getQuality().then(setData).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  async function run(name, start) {
    setBusy(name); setError("");
    try {
      const { job_id } = await start();
      const job = await watchQualityJob(job_id, (j) => setBusy(`${name} ${j.done}/${j.total || "…"}`));
      if (job.status === "failed") setError(job.error);
      await refresh();
    } catch (e) { setError(e.message); }
    setBusy("");
  }

  async function mark(item, correct) {
    const next = item.human === correct ? null : correct; // clicking the same verdict again clears it
    await reviewArtifact(item.id, next).catch((e) => setError(e.message));
    setData({ ...data, artifacts: { ...data.artifacts, items: data.artifacts.items.map((i) => (i.id === item.id ? { ...i, human: next } : i)) } });
    refresh();
  }

  const artifacts = data.artifacts, answers = data.answers;
  const items = (artifacts?.items ?? []).filter((i) => i.type === kind && (show === "all" || (show === "unreviewed" ? i.human === null : i.llm_supported === false)));
  const summary = artifacts?.summary?.[kind];

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Quality checks</h1>
      <p className="hint mb-5 mt-1">The LLM writes Q&A pairs, factoids and answers, and can be wrong. These checks measure how often.</p>
      {error && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}

      <h2 className="mb-1 text-lg font-semibold">Answers</h2>
      <p className="hint mb-3">Each test question is answered as the Ask page does, then an LLM grades the answer against the passages it was given. Off-topic questions (<code>eval/unanswerable.json</code>) should get “I couldn’t find this”.</p>
      <button className="btn !py-2" disabled={!!busy} onClick={() => run("Answering", () => runAnswerEval("test"))}>{busy.startsWith("Answering") ? busy : "Evaluate answers"}</button>
      {answers && (
        <>
          <div className="my-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Tile label="Faithfulness" value={pct(answers.faithfulness)} hint="claims supported by the context" />
            <Tile label="Relevance" value={pct(answers.relevance)} hint="answers the question" />
            <Tile label="Wrongly abstained" value={pct(answers.wrongly_abstained)} hint={`of ${answers.answerable} answerable`} />
            <Tile label="Correctly abstained" value={pct(answers.correctly_abstained)} hint={`of ${answers.unanswerable} off-topic`} />
          </div>
          {answers.failed > 0 && <p className="hint">{answers.failed} questions failed (for example a rate limit) and are left out of the rates.</p>}
          <details className="mb-6">
            <summary className="cursor-pointer text-sm text-accent">Every question</summary>
            <ul className="mt-2 divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel text-sm">
              {answers.rows.map((r) => (
                <li key={r.id} className="px-4 py-2">
                  <p className="font-medium">{r.question} <span className="text-xs font-normal text-muted">{r.answerable ? "answerable" : "off-topic"}</span></p>
                  <p className="text-muted">{r.abstained === null ? r.reason : r.abstained ? "Abstained." : r.answer}</p>
                  {r.abstained === false && <p className={r.faithful && r.relevant ? "text-ok" : "text-bad"}>{r.faithful ? "faithful" : "not faithful"} · {r.relevant ? "relevant" : "not relevant"} — {r.reason}</p>}
                </li>
              ))}
            </ul>
          </details>
        </>
      )}

      <h2 className="mb-1 mt-8 text-lg font-semibold">Generated Q&A pairs and factoids</h2>
      <p className="hint mb-3">A random sample is checked against the chunk each came from. Read the source yourself and mark each item right or wrong: your error rate is the one to report; the LLM check only helps you find the wrong ones.</p>
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-sm text-muted">Items per kind <input type="number" min="1" max="1000" className="field !inline-block !w-24" value={sample} onChange={(e) => setSample(Number(e.target.value) || 1)} /></label>
        <button className="btn !py-2" disabled={!!busy} onClick={() => run("Checking", () => runArtifactCheck(sample))}>{busy.startsWith("Checking") ? busy : "Check a sample"}</button>
      </div>
      {artifacts && (
        <>
          <div className="my-3 flex flex-wrap items-center gap-2">
            <select className="field !w-auto" value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Kind">{Object.entries(KIND).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>
            <select className="field !w-auto" value={show} onChange={(e) => setShow(e.target.value)} aria-label="Show">
              <option value="unreviewed">Not reviewed yet</option><option value="flagged">Flagged by the LLM</option><option value="all">All</option>
            </select>
          </div>
          {summary && (
            <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
              <Tile label="Checked by the LLM" value={summary.checked} />
              <Tile label="LLM-flagged error rate" value={pct(summary.llm_error_rate)} hint={`${summary.llm_errors} flagged`} />
              <Tile label="Reviewed by you" value={summary.reviewed} />
              <Tile label="Your error rate" value={pct(summary.human_error_rate)} hint={summary.agreement === null ? "" : `agrees with the LLM ${pct(summary.agreement)}`} />
            </div>
          )}
          <ul className="space-y-3">
            {items.slice(0, 50).map((i) => (
              <li key={i.id} className="card !my-0">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">Generated</p>
                    {i.question && <p className="font-medium">{i.question}</p>}
                    <p>{i.answer}</p>
                    <p className={`mt-2 text-sm ${i.llm_supported ? "text-ok" : "text-bad"}`}>LLM check: {i.llm_supported === null ? "failed" : i.llm_supported ? "supported" : "not supported"} — {i.reason}</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">Source chunk{i.section ? ` · ${i.section}` : ""}{i.page ? ` · p. ${i.page}` : ""}</p>
                    <p className="text-sm">{i.source}</p>
                  </div>
                </div>
                <div className="mt-3 flex gap-2">
                  <button className={`chip !mr-0 cursor-pointer ${i.human === true ? "!border-ok text-ok" : ""}`} onClick={() => mark(i, true)}><Check size={14} className="mr-1 inline" />Correct</button>
                  <button className={`chip !mr-0 cursor-pointer ${i.human === false ? "!border-bad text-bad" : ""}`} onClick={() => mark(i, false)}><X size={14} className="mr-1 inline" />Incorrect</button>
                </div>
              </li>
            ))}
          </ul>
          {items.length === 0 && <p className="hint rounded-2xl border border-dashed border-line p-8 text-center">Nothing to show here.</p>}
          {items.length > 50 && <p className="hint mt-2">Showing the first 50 of {items.length}.</p>}
        </>
      )}
    </div>
  );
}
