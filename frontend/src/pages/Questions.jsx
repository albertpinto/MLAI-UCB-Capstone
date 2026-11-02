import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Pencil, Plus, Save, Trash2, Upload } from "lucide-react";
import { checkEvalQuestions, getEvalQuestions, saveEvalQuestions } from "../api.js";

const BLANK = { id: "", question: "", evidence: [], split: "test", note: "" };

export default function Questions() {
  const [saved, setSaved] = useState([]); // what is on disk
  const [questions, setQuestions] = useState([]); // the draft being edited
  const [checks, setChecks] = useState({}); // id -> sections whose text contains the question's evidence
  const [editing, setEditing] = useState(null); // index of the row being edited, or "new"
  const [filter, setFilter] = useState("");
  const [split, setSplit] = useState("all");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const file = useRef(null);

  useEffect(() => { getEvalQuestions().then((d) => { setSaved(d.questions); setQuestions(d.questions); }).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    if (!questions.length) { setChecks({}); return; }
    const timer = setTimeout(() => checkEvalQuestions(questions).then((d) => setChecks(d.checks)).catch(() => {}), 400);
    return () => clearTimeout(timer);
  }, [questions]);

  const dirty = JSON.stringify(saved) !== JSON.stringify(questions);
  const shown = useMemo(() => questions.map((q, i) => ({ q, i })).filter(({ q }) =>
    (split === "all" || q.split === split) && `${q.id} ${q.question} ${q.evidence.join(" ")}`.toLowerCase().includes(filter.trim().toLowerCase())), [questions, split, filter]);
  const counts = { test: questions.filter((q) => q.split === "test").length, train: questions.filter((q) => q.split === "train").length };
  const unmatched = questions.filter((q) => checks[q.id]?.length === 0).length;

  function commit(q) {
    const id = q.id.trim();
    const evidence = q.evidence.map((e) => e.trim()).filter(Boolean);
    if (!id || !q.question.trim() || !evidence.length) return setError("A question needs an id, the question text and at least one evidence phrase.");
    if (questions.some((o, i) => o.id === id && i !== editing)) return setError(`The id "${id}" is already used.`);
    const next = { ...q, id, question: q.question.trim(), evidence };
    setQuestions(editing === "new" ? [...questions, next] : questions.map((o, i) => (i === editing ? next : o)));
    setEditing(null);
    setError("");
  }

  async function save() {
    setError(""); setMessage("");
    try {
      await saveEvalQuestions(questions);
      setSaved(questions);
      setMessage(`Saved ${questions.length} questions.`);
    } catch (e) { setError(e.message); }
  }

  async function importFile(e) {
    const chosen = e.target.files[0];
    e.target.value = "";
    if (!chosen) return;
    try {
      const list = JSON.parse(await chosen.text());
      if (!Array.isArray(list) || !list.every((q) => q && q.id && q.question && Array.isArray(q.evidence))) throw new Error("Expected a list of questions with id, question and evidence.");
      setQuestions(list.map((q) => ({ ...BLANK, ...q })));
      setMessage(`Loaded ${list.length} questions from ${chosen.name}. Save to keep them.`);
      setError("");
    } catch (err) { setError(`Could not import: ${err.message}`); }
  }

  function exportFile() {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(questions, null, 1)], { type: "application/json" }));
    link.download = "questions.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Evaluation questions</h1>
      <p className="hint mb-5 mt-1">
        Each question lists <em>evidence</em>: words copied from the passage that answers it. A section that contains one is the right answer, so the labels survive a re-chunk.
        Questions marked <b>train</b> may be used to fine-tune; the Parts that hold a <b>test</b> question are kept out of fine-tuning.
      </p>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input className="field !w-64" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter questions" aria-label="Filter questions" />
        <select className="field !w-auto" value={split} onChange={(e) => setSplit(e.target.value)} aria-label="Split">
          <option value="all">All ({questions.length})</option>
          <option value="test">Test ({counts.test})</option>
          <option value="train">Train ({counts.train})</option>
        </select>
        <span className="ml-auto flex flex-wrap items-center gap-2">
          <button className="btn !bg-none !bg-panel !text-ink border border-line inline-flex items-center gap-2 !py-2" onClick={() => file.current.click()}><Upload size={16} />Import</button>
          <input ref={file} type="file" accept="application/json,.json" hidden onChange={importFile} />
          <button className="btn !bg-none !bg-panel !text-ink border border-line inline-flex items-center gap-2 !py-2" onClick={exportFile} disabled={!questions.length}><Download size={16} />Export</button>
          <button className="btn !bg-none !bg-panel !text-ink border border-line inline-flex items-center gap-2 !py-2" onClick={() => setEditing("new")}><Plus size={16} />Add</button>
          <button className="btn inline-flex items-center gap-2 !py-2" onClick={save} disabled={!dirty}><Save size={16} />Save{dirty ? " changes" : "d"}</button>
        </span>
      </div>
      {unmatched > 0 && <p className="mb-3 text-sm text-bad" role="status">{unmatched} question{unmatched > 1 ? "s have" : " has"} evidence that matches no section, so an evaluation would leave {unmatched > 1 ? "them" : "it"} out.</p>}
      {message && <p className="mb-3 text-ok" role="status">{message}</p>}
      {error && <p className="mb-3 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}

      {editing === "new" && <Editor initial={BLANK} onSave={commit} onCancel={() => { setEditing(null); setError(""); }} />}
      <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel shadow-card">
        {shown.map(({ q, i }) => editing === i ? (
          <li key={q.id} className="p-4"><Editor initial={q} onSave={commit} onCancel={() => { setEditing(null); setError(""); }} /></li>
        ) : (
          <li key={q.id} className="flex items-start gap-3 px-4 py-3">
            <span className="min-w-0 flex-1">
              <span className="block font-medium">{q.question} <span className="text-xs font-normal text-muted">{q.id}</span></span>
              <span className="block text-sm text-muted">Evidence: {q.evidence.map((e) => `“${e}”`).join(", ")}</span>
              <span className="block text-xs text-muted">
                {checks[q.id] === undefined ? "" : checks[q.id].length ? `Found in: ${checks[q.id].slice(0, 2).join("; ")}${checks[q.id].length > 2 ? ` and ${checks[q.id].length - 2} more` : ""}` : <span className="text-bad">Not found in any section ✗</span>}
              </span>
            </span>
            <button className={`chip !mr-0 cursor-pointer ${q.split === "train" ? "!border-accent text-accent" : ""}`} title="Switch between train and test"
              onClick={() => setQuestions(questions.map((o, j) => (j === i ? { ...o, split: o.split === "train" ? "test" : "train" } : o)))}>{q.split}</button>
            <button className="rounded-lg p-1.5 text-muted hover:bg-soft hover:text-accent" aria-label={`Edit ${q.id}`} onClick={() => { setEditing(i); setError(""); }}><Pencil size={16} /></button>
            <button className="rounded-lg p-1.5 text-muted hover:bg-soft hover:text-bad" aria-label={`Delete ${q.id}`}
              onClick={() => window.confirm(`Delete question ${q.id}?`) && setQuestions(questions.filter((_, j) => j !== i))}><Trash2 size={16} /></button>
          </li>
        ))}
      </ul>
      {shown.length === 0 && <p className="hint rounded-2xl border border-dashed border-line p-8 text-center">No questions to show.</p>}
    </div>
  );
}

function Editor({ initial, onSave, onCancel }) {
  const [q, setQ] = useState({ ...initial, evidence: initial.evidence.join("\n") });
  const set = (k) => (e) => setQ({ ...q, [k]: e.target.value });
  return (
    <form className="card space-y-3 !my-0" onSubmit={(e) => { e.preventDefault(); onSave({ ...q, evidence: q.evidence.split("\n") }); }}>
      <div className="flex gap-3">
        <input className="field !w-48" value={q.id} onChange={set("id")} placeholder="Id, e.g. q63" aria-label="Id" />
        <select className="field !w-auto" value={q.split} onChange={set("split")} aria-label="Split"><option>test</option><option>train</option></select>
      </div>
      <input className="field" value={q.question} onChange={set("question")} placeholder="The question, in everyday words" aria-label="Question" />
      <textarea className="field" rows={3} value={q.evidence} onChange={set("evidence")} placeholder="Evidence: one phrase per line, copied from the passage that answers it" aria-label="Evidence" />
      <input className="field" value={q.note} onChange={set("note")} placeholder="Note (optional), e.g. who wrote the question" aria-label="Note" />
      <div className="flex gap-2">
        <button className="btn !py-2">Done</button>
        <button type="button" className="rounded-lg border border-line px-4 py-2 text-muted hover:text-ink" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}
