import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRight, Loader2, Network, Search } from "lucide-react";
import { findEntities, getEntity, getGraphStats, rebuildGraph } from "../api.js";
import { Chart } from "../charts.jsx";

// The knowledge graph has thousands of entities, so it is explored one entity at a time: search or pick a hub, see it and its
// closest neighbours as a network, and read the triplets (with their source passages) that the picture comes from.
export default function Graph() {
  const [params, setParams] = useSearchParams();
  const selected = params.get("entity");
  const [stats, setStats] = useState(null);
  const [entity, setEntity] = useState(null);
  const [error, setError] = useState("");
  const [building, setBuilding] = useState(false);

  const load = () => getGraphStats().then(setStats).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!stats?.built) return;
    const id = selected ?? stats.top_entities[0]?.id; // open on the most connected entity
    if (!id) return;
    setError("");
    getEntity(id).then(setEntity).catch((e) => setError(e.message));
  }, [stats, selected]);

  const open = (id) => setParams({ entity: id });

  async function build() {
    setBuilding(true);
    setError("");
    try { await rebuildGraph(); await load(); } catch (e) { setError(e.message); } finally { setBuilding(false); }
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Knowledge graph</h1>
      <p className="hint mb-5 mt-1">
        Entities and relations the LLM found in each passage. Spellings are merged ("the Governors" and "Governor" are one entity). Pick an entity to see how it connects;
        every relation links back to the passage it came from.
      </p>
      {error && <p className="mb-4 rounded-xl border border-bad/30 bg-bad/10 p-3 text-bad" role="alert">{error}</p>}
      {stats && !stats.built && (
        <div className="card text-center">
          <Network className="mx-auto mb-2 text-accent" />
          <p className="mb-3">The graph has not been built yet. It is made from the triplets already stored, so no new LLM calls are needed.</p>
          <button className="btn inline-flex items-center gap-2" onClick={build} disabled={building}>{building && <Loader2 size={16} className="animate-spin" />}Build the graph</button>
        </div>
      )}
      {stats?.built && (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Tile label="Entities" value={stats.entities} />
            <Tile label="Relations" value={stats.relations} />
            <Tile label="Documents" value={stats.documents} />
            <button className="card my-0 text-left text-sm text-muted hover:text-accent" onClick={build} disabled={building} title="Rebuild from the stored triplets">
              {building ? "Rebuilding…" : "Rebuild the graph"}<span className="hint block">Runs automatically after each ingest.</span>
            </button>
          </div>
          <Finder onPick={open} />
          {entity && <Entity entity={entity} onOpen={open} />}
          <Overview stats={stats} onOpen={open} />
        </>
      )}
    </div>
  );
}

function Tile({ label, value }) {
  return (
    <div className="card my-0 min-w-0">
      <p className="hint truncate">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{value.toLocaleString()}</p>
    </div>
  );
}

function Finder({ onPick }) {
  const [text, setText] = useState("");
  const [found, setFound] = useState([]);
  useEffect(() => {
    if (text.trim().length < 2) { setFound([]); return; }
    const timer = setTimeout(() => findEntities(text).then((d) => setFound(d.entities)).catch(() => setFound([])), 250);
    return () => clearTimeout(timer);
  }, [text]);
  return (
    <div className="relative mb-4">
      <Search size={16} className="pointer-events-none absolute left-3.5 top-3.5 text-muted" />
      <input className="field !pl-10" value={text} onChange={(e) => setText(e.target.value)} placeholder="Find an entity, e.g. Governor, High Court, Lok Sabha" aria-label="Find an entity" />
      {found.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-xl border border-line bg-panel shadow-card">
          {found.map((f) => (
            <li key={f.id}><button className="flex w-full items-center justify-between px-4 py-2 text-left hover:bg-soft" onClick={() => { onPick(f.id); setText(""); setFound([]); }}>
              <span>{f.label}</span><span className="text-xs text-muted">{f.degree} relations</span>
            </button></li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Entity({ entity, onOpen }) {
  const { nodes, edges } = entity;
  const at = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const label = (n) => (n.x > 0.3 ? "middle right" : n.x < -0.3 ? "middle left" : n.y > 0 ? "top center" : "bottom center");
  const line = { x: [], y: [] };
  edges.forEach((e) => { line.x.push(at[e.source].x, at[e.target].x, null); line.y.push(at[e.source].y, at[e.target].y, null); });
  const middles = edges.map((e) => ({ x: (at[e.source].x + at[e.target].x) / 2, y: (at[e.source].y + at[e.target].y) / 2, text: `${at[e.source].label} → ${e.relations.join(" / ")} → ${at[e.target].label}${e.count > 1 ? ` (${e.count} times)` : ""}` }));
  const group = (centre) => nodes.filter((n) => n.centre === centre);

  return (
    <section className="card !my-0 mb-4">
      <h2 className="font-semibold">{entity.label}</h2>
      <p className="hint">{entity.nodes.length - 1} of {entity.neighbours_total} neighbours shown, the most connected first. Select an entity to move to it.</p>
      <Chart
        height={480}
        label={`Network around ${entity.label}: ${nodes.length - 1} neighbours`}
        onClick={(e) => { const id = e.points?.[0]?.customdata; if (id && id !== entity.id) onOpen(id); }}
        data={(t) => [
          { type: "scatter", mode: "lines", x: line.x, y: line.y, line: { color: t.muted, width: 1 }, opacity: 0.45, hoverinfo: "skip", showlegend: false },
          { type: "scatter", mode: "markers", x: middles.map((m) => m.x), y: middles.map((m) => m.y), text: middles.map((m) => m.text), hoverinfo: "text", marker: { size: 10, color: t.muted, opacity: 0 }, showlegend: false },
          { type: "scatter", mode: "markers+text", name: "Related entities", x: group(false).map((n) => n.x), y: group(false).map((n) => n.y), text: group(false).map((n) => n.label), textposition: group(false).map(label),
            textfont: { color: t.text, size: 11 }, customdata: group(false).map((n) => n.id), hovertemplate: "%{text}<extra>%{customdata}</extra>",
            marker: { size: group(false).map((n) => 10 + Math.min(Math.sqrt(n.degree) * 2, 16)), color: t.s2, line: { width: 2, color: t.panel } } },
          { type: "scatter", mode: "markers+text", name: "Selected entity", x: [0], y: [0], text: [entity.label], textposition: "top center", textfont: { color: t.text, size: 13 }, customdata: [entity.id],
            hovertemplate: "%{text}<extra></extra>", marker: { size: 26, color: t.s1, line: { width: 2, color: t.panel } } },
        ]}
        layout={() => ({ margin: { l: 16, r: 16, t: 16, b: 40 }, xaxis: { visible: false, range: [-2, 2] }, yaxis: { visible: false, range: [-1.4, 1.4] }, hovermode: "closest" })}
      />
      <h3 className="mb-2 mt-4 text-sm font-semibold">Relations of {entity.label} ({entity.facts.length}{entity.facts.length === 40 ? "+" : ""})</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted"><tr><th className="w-[45%] py-2 pr-3 font-medium">Relation</th><th className="py-2 pr-3 font-medium">Where it was stated</th><th /></tr></thead>
          <tbody>
            {entity.facts.map((f, i) => (
              <tr key={i} className="border-t border-line align-top">
                <td className="py-2 pr-3">
                  <button className="text-accent hover:underline" onClick={() => onOpen(f.subject_id)}>{f.subject}</button>{" "}
                  <span className="text-muted">{f.relation}</span>{" "}
                  <button className="text-accent hover:underline" onClick={() => onOpen(f.object_id)}>{f.object}</button>
                </td>
                <td className="py-2 pr-3 text-xs text-muted">
                  <span className="block">{[f.section, f.page ? `page ${f.page}` : ""].filter(Boolean).join(" · ")}</span>
                  <span className="line-clamp-2">{f.text}</span>
                </td>
                <td className="py-2"><Link className="inline-flex items-center gap-1 whitespace-nowrap text-xs font-medium text-accent hover:underline" to={`/passage/${f.document_id}/${f.parent_id}`}>Passage <ArrowRight size={12} /></Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Overview({ stats, onOpen }) {
  const bars = (rows, name, value, click) => (t) => [{
    type: "bar", orientation: "h", x: rows.map((r) => r[value]).reverse(), y: rows.map((r) => r[name]).reverse(), text: rows.map((r) => r[value]).reverse(), textposition: "outside", cliponaxis: false,
    customdata: rows.map((r) => r.id).reverse(), marker: { color: t.s1, line: { width: 2, color: t.panel } }, hovertemplate: "%{y}: %{x}<extra>" + click + "</extra>",
  }];
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="card !my-0">
        <h2 className="font-semibold">Most connected entities</h2>
        <p className="hint">Select a bar to open that entity.</p>
        <Chart height={420} label="Most connected entities, by number of relations" data={bars(stats.top_entities, "label", "degree", "relations")}
          onClick={(e) => { const id = e.points?.[0]?.customdata; if (id) onOpen(id); }}
          layout={() => ({ margin: { l: 150, r: 40, t: 8, b: 40 }, xaxis: { title: { text: "relations" } }, yaxis: { automargin: true } })} />
      </section>
      <section className="card !my-0">
        <h2 className="font-semibold">Most common relations</h2>
        <p className="hint">The verbs the LLM used between entities.</p>
        <Chart height={420} label="Most common relations, by count" data={bars(stats.top_relations, "relation", "count", "times")}
          layout={() => ({ margin: { l: 150, r: 40, t: 8, b: 40 }, xaxis: { title: { text: "times used" } }, yaxis: { automargin: true } })} />
      </section>
    </div>
  );
}
