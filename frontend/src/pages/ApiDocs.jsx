import { useEffect, useState } from "react";
import { getOpenApi } from "../api.js";

const TAG_ORDER = ["Ingest", "Search", "Documents", "System"];
const METHOD_STYLE = {
  get: "bg-blue-600/15 text-blue-600 dark:text-blue-400",
  post: "bg-emerald-600/15 text-emerald-700 dark:text-emerald-400",
  delete: "bg-red-600/15 text-red-600 dark:text-red-400",
};
// where the API itself lives (the UI only proxies /api in development)
const API = `${location.protocol}//${location.hostname}:8000`;

const deref = (schema, spec) => (schema?.$ref ? spec.components.schemas[schema.$ref.split("/").pop()] : schema);

function typeOf(s, spec) {
  s = deref(s, spec) ?? {};
  if (s.anyOf) return typeOf(s.anyOf.find((x) => x.type !== "null"), spec);
  if (s.type === "array") return `${typeOf(s.items, spec)}[]`;
  return s.format === "binary" ? "file" : s.type ?? "any";
}

function fields(body, spec) {
  const schema = deref(Object.values(body.content)[0].schema, spec);
  return Object.entries(schema.properties ?? {}).map(([name, p]) => ({
    name, type: typeOf(p, spec), required: schema.required?.includes(name), description: p.description ?? "", example: p.examples?.[0] ?? p.default,
  }));
}

function curlFor(path, method, op, spec) {
  const url = `${API}${path.replace(/\{(\w+)\}/g, "<$1>")}`;
  let cmd = `curl${method === "get" ? "" : ` -X ${method.toUpperCase()}`} ${url}`;
  const body = op.requestBody;
  if (body?.content["multipart/form-data"]) cmd += ` \\\n  -F "files=@guide.txt"`;
  else if (body) {
    const sample = Object.fromEntries(fields(body, spec).filter((f) => f.required || f.example !== undefined).map((f) => [f.name, f.example ?? (f.type === "string" ? "…" : null)]));
    cmd += ` \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(sample)}'`;
  }
  return cmd;
}

// `code` spans and blank-line paragraphs, enough for the docstrings
const Prose = ({ text }) => (
  <div className="space-y-2 text-sm leading-relaxed">
    {text.split(/\n\s*\n/).map((para, i) => (
      <p key={i}>{para.replace(/\n/g, " ").split(/(`[^`]+`)/).map((t, j) => t.startsWith("`") ? <code key={j} className="rounded bg-bg px-1 py-0.5 text-[.85em]">{t.slice(1, -1)}</code> : t)}</p>
    ))}
  </div>
);

function CodeBlock({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-lg bg-nav p-3 pr-16 text-xs leading-relaxed text-slate-200">{text}</pre>
      <button onClick={copy} className="absolute right-2 top-2 rounded bg-white/10 px-2 py-1 text-xs text-slate-200 hover:bg-white/20">{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

const Table = ({ rows }) => (
  <table className="w-full rounded-lg border border-line text-sm">
    <thead className="bg-bg text-left text-xs uppercase tracking-wider text-muted"><tr className="[&>th]:px-3 [&>th]:py-2"><th>Name</th><th>Type</th><th>Description</th></tr></thead>
    <tbody>
      {rows.map((r) => (
        <tr key={r.name} className="border-t border-line align-top [&>td]:px-3 [&>td]:py-2">
          <td><code>{r.name}</code>{r.required && <span className="ml-1.5 text-xs text-bad">required</span>}</td>
          <td className="text-muted">{r.type}</td>
          <td>{r.description}</td>
        </tr>
      ))}
    </tbody>
  </table>
);

function Endpoint({ path, method, op, spec }) {
  const params = (op.parameters ?? []).map((p) => ({ name: p.name, type: `${typeOf(p.schema, spec)} (${p.in})`, required: p.required, description: p.description ?? "" }));
  const body = op.requestBody ? fields(op.requestBody, spec) : [];
  const example = op.responses?.["200"]?.content?.["application/json"]?.example;
  const Label = ({ children }) => <h4 className="mb-1.5 mt-4 text-xs font-semibold uppercase tracking-wider text-muted">{children}</h4>;
  return (
    <details id={`${method}-${path}`} className="card group scroll-mt-20 !my-3">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-3 gap-y-1">
        <span className={`w-16 rounded-md py-0.5 text-center text-xs font-bold uppercase ${METHOD_STYLE[method]}`}>{method}</span>
        <code className="font-semibold">{path}</code>
        <span className="text-sm text-muted sm:ml-auto">{op.summary}</span>
      </summary>
      <div className="mt-3 border-t border-line pt-3">
        <Prose text={op.description ?? ""} />
        {params.length > 0 && <><Label>Path parameters</Label><Table rows={params} /></>}
        {body.length > 0 && <><Label>Request body ({Object.keys(op.requestBody.content)[0]})</Label><Table rows={body} /></>}
        <Label>Example request</Label>
        <CodeBlock text={curlFor(path, method, op, spec)} />
        {example && <><Label>Example response</Label><CodeBlock text={JSON.stringify(example, null, 2)} /></>}
      </div>
    </details>
  );
}

export default function ApiDocs() {
  const [spec, setSpec] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");

  useEffect(() => { getOpenApi().then(setSpec).catch((e) => setError(e.message)); }, []);
  if (error) return <p className="text-bad" role="alert">{error}</p>;
  if (!spec) return <p className="hint">Loading…</p>;

  const q = filter.trim().toLowerCase();
  const ops = Object.entries(spec.paths).flatMap(([path, methods]) => Object.entries(methods).map(([method, op]) => ({ path, method, op })))
    .filter(({ path, op }) => !q || `${path} ${op.summary} ${op.description}`.toLowerCase().includes(q));
  const groups = TAG_ORDER.map((tag) => [tag, ops.filter((o) => o.op.tags?.[0] === tag)]).filter(([, list]) => list.length);

  return (
    <>
      <h1 className="text-2xl font-semibold tracking-tight">API reference</h1>
      <p className="hint mb-4 mt-1">{spec.info.description} Base URL: <code>{API}</code> · version {spec.info.version}</p>
      <div className="mb-2 flex flex-wrap gap-2">
        <input className="field flex-1" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter endpoints, e.g. search or upload" />
        <a className="btn inline-block no-underline" href={`${API}/docs`} target="_blank" rel="noreferrer">Try it in Swagger ↗</a>
      </div>
      {groups.length === 0 && <p className="hint">No endpoints match.</p>}
      {groups.map(([tag, list]) => (
        <section key={tag}>
          <h2 className="mb-1 mt-7 text-lg font-semibold">{tag}</h2>
          {list.map((o) => <Endpoint key={o.method + o.path} {...o} spec={spec} />)}
        </section>
      ))}
    </>
  );
}
