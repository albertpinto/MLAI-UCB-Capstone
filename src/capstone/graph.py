"""The knowledge graph: entities and relations from the triplets the LLM extracted per chunk.

Nodes are normalised entities (case, plurals, articles and aliases merge into one node); every edge remembers the chunk it
was stated in. At query time the entities named in the question are looked up, and the chunks their edges came from join the
candidates (graph-assisted retrieval). This is deliberately NOT Graph RAG: no communities, no graph summaries by an LLM, no router.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import networkx as nx

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "such", "its", "their", "his", "her", "any", "each", "every", "all"}
MAX_LINK_WORDS = 6  # the longest entity name looked for in a question
SPECIFIC_DEGREE = 25  # an entity with more relations than this ("President", "State") names too many chunks to say anything alone
STOP = {"is", "are", "be", "of", "to", "by", "for", "in", "as", "on", "at", "or", "and", "a", "an", "the", "has", "have", "with", "from", "not", "do", "does"}
RRF_K = 60  # reciprocal rank fusion constant, shared with the evaluation
GRAPH_WEIGHT = 0.1  # how much the graph's ranking counts against the dense one when fused. Swept on the train questions: 0.1 never lowered them, 1.0 pushed good dense hits out


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _stem(word: str) -> str:
    word = _singular(word)
    return word[:-3] if len(word) > 5 and word.endswith("ing") else word[:-2] if len(word) > 4 and word.endswith("ed") else word


def normalise(name: str, aliases: dict[str, str] | None = None) -> str:
    """"The Governors' powers" and "governor power" are the same entity: lower case, no possessives, articles or plural endings."""
    words = re.findall(r"[a-z0-9]+", re.sub(r"['’]s\b", "", name.lower()))
    while words and words[0] in DETERMINERS:
        words.pop(0)
    key = " ".join(_singular(w) for w in words)
    return (aliases or {}).get(key, key)


def load_aliases(path: Path | None) -> dict[str, str]:
    """Optional JSON file {"alias": "canonical name"}; both sides are normalised, so "SC" -> "Supreme Court" merges those nodes."""
    if not path or not path.is_file():
        return {}
    return {normalise(a): normalise(c) for a, c in json.loads(path.read_text()).items()}


def _display(name: str) -> str:
    """The name as a label: "the President" -> "President"."""
    words = name.split()
    while len(words) > 1 and words[0].lower() in DETERMINERS:
        words.pop(0)
    return " ".join(words)


def build(triplets: list[dict], aliases: dict[str, str] | None = None) -> nx.MultiDiGraph:
    """`triplets` are triplet payloads: subject, relation, object, document_id, parent_id, chunk_id."""
    graph = nx.MultiDiGraph()
    surface: dict[str, Counter] = {}  # node -> how each spelling was written, to label the node with the commonest
    seen = set()
    for t in triplets:
        s, o = normalise(t["subject"], aliases), normalise(t["object"], aliases)
        if not s or not o or s == o:
            continue
        edge = (s, t["relation"].strip().lower(), o, t["document_id"], t["chunk_id"])
        if edge in seen:
            continue
        seen.add(edge)
        for key, name in ((s, t["subject"]), (o, t["object"])):
            surface.setdefault(key, Counter())[_display(name)] += 1
        graph.add_edge(s, o, relation=edge[1], document_id=t["document_id"], parent_id=t["parent_id"], chunk_id=t["chunk_id"])
    for key, names in surface.items():
        graph.nodes[key]["label"] = names.most_common(1)[0][0]
    return graph


def save(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False))


def load(path: Path) -> nx.MultiDiGraph | None:
    return nx.node_link_graph(json.loads(path.read_text()), edges="edges") if path.is_file() else None


def link_entities(graph: nx.MultiDiGraph, query: str, aliases: dict[str, str] | None = None) -> list[str]:
    """Entities named in the question: every run of its words that is a node ("governor" and "governor of a state" both count)."""
    words = [_singular(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"['’]s\b", "", query.lower()))]
    found = []
    for size in range(min(MAX_LINK_WORDS, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            key = " ".join(words[start:start + size])
            key = (aliases or {}).get(key, key)
            if key in graph and key not in found:
                found.append(key)
    return found


def graph_chunks(graph: nx.MultiDiGraph, entities: list[str], query: str, limit: int = 20) -> list[tuple[tuple[str, int], list[str]]]:
    """Chunks whose triplets touch the question's entities, best first, each with the entities (labels) that led to it.

    An edge ties an entity to a neighbour, so the chunk that stated it is about both. A rare entity says more than a common one,
    so it counts for more; an edge also counts for more when its other end is in the question too or its relation is what the
    question asks ("appoints"). An edge of a very common entity ("President") is used only in those two cases.
    """
    asked = {_stem(w) for w in re.findall(r"[a-z0-9]+", query.lower())} - STOP
    score: dict[tuple[str, int], float] = {}
    via: dict[tuple[str, int], list[str]] = {}
    for entity in entities:
        weight = 1 / math.log2(2 + graph.degree(entity))
        label = graph.nodes[entity].get("label", entity)
        for a, b, data in list(graph.out_edges(entity, data=True)) + list(graph.in_edges(entity, data=True)):
            tied = (b if a == entity else a) in entities
            said = bool({_stem(w) for w in data["relation"].split()} - STOP & asked)
            if graph.degree(entity) > SPECIFIC_DEGREE and not (tied or said):
                continue
            chunk = (data["document_id"], data["chunk_id"])
            score[chunk] = score.get(chunk, 0.0) + weight * (1 + tied + said)
            if label not in via.setdefault(chunk, []):
                via[chunk].append(label)
    return [(chunk, via[chunk]) for chunk in sorted(score, key=lambda c: -score[c])[:limit]]


def search_entities(graph: nx.MultiDiGraph, text: str, limit: int = 15) -> list[dict]:
    needle = normalise(text)
    hits = [n for n in graph if needle and needle in n]
    hits.sort(key=lambda n: (not n.startswith(needle), -graph.degree(n), n))
    return [{"id": n, "label": graph.nodes[n].get("label", n), "degree": graph.degree(n)} for n in hits[:limit]]


def stats(graph: nx.MultiDiGraph, top: int = 15) -> dict:
    hubs = sorted(graph, key=lambda n: -graph.degree(n))[:top]
    relations = Counter(d["relation"] for _, _, d in graph.edges(data=True))
    return {
        "entities": graph.number_of_nodes(), "relations": graph.number_of_edges(),
        "documents": len({d["document_id"] for _, _, d in graph.edges(data=True)}),
        "top_entities": [{"id": n, "label": graph.nodes[n]["label"], "degree": graph.degree(n)} for n in hubs],
        "top_relations": [{"relation": r, "count": c} for r, c in relations.most_common(top)],
    }


def neighbourhood(graph: nx.MultiDiGraph, entity: str, max_neighbours: int = 20) -> dict:
    """The entity, its most-connected neighbours on a circle around it, and the relations between all of them (parallel edges merged).
    Positions are computed here so the browser only draws."""
    ties = Counter(m for a, b in list(graph.in_edges(entity)) + list(graph.out_edges(entity)) for m in (a, b) if m != entity)
    neighbours = [n for n, _ in ties.most_common(max_neighbours)]
    nodes = [{"id": entity, "label": graph.nodes[entity]["label"], "degree": graph.degree(entity), "x": 0.0, "y": 0.0, "centre": True}]
    for i, n in enumerate(neighbours):
        angle = 2 * math.pi * i / len(neighbours)
        nodes.append({"id": n, "label": graph.nodes[n]["label"], "degree": graph.degree(n), "x": math.cos(angle), "y": math.sin(angle), "centre": False})
    shown = {n["id"] for n in nodes}
    merged: dict[tuple[str, str], Counter] = {}
    for a, b, d in graph.edges(data=True):
        if a in shown and b in shown:
            merged.setdefault((a, b), Counter())[d["relation"]] += 1
    edges = [{"source": a, "target": b, "relations": [r for r, _ in c.most_common(3)], "count": sum(c.values())} for (a, b), c in merged.items()]
    return {"nodes": nodes, "edges": edges, "neighbours_total": len(ties)}


def facts(graph: nx.MultiDiGraph, entity: str, limit: int = 40) -> list[dict]:
    """The triplets that touch an entity (as subject or object), with the chunk each was stated in."""
    rows = [
        {"subject": graph.nodes[a]["label"], "subject_id": a, "relation": d["relation"], "object": graph.nodes[b]["label"], "object_id": b,
         "document_id": d["document_id"], "parent_id": d["parent_id"], "chunk_id": d["chunk_id"]}
        for a, b, d in list(graph.out_edges(entity, data=True)) + list(graph.in_edges(entity, data=True))
    ]
    return rows[:limit]


def summary_text(graph: nx.MultiDiGraph, title: str, document_id: str, top: int = 8) -> str:
    """A short description of one document's part of the graph, deterministic (no LLM, so nothing to get wrong). It is stored as a vector point."""
    mine = [(a, b, d) for a, b, d in graph.edges(data=True) if d["document_id"] == document_id]
    if not mine:
        return ""
    degree = Counter(n for a, b, _ in mine for n in (a, b))
    entities = [graph.nodes[n]["label"] for n, _ in degree.most_common(top)]
    relations = [r for r, _ in Counter(d["relation"] for _, _, d in mine).most_common(5)]
    return f"Knowledge graph of {title}: {len(degree)} entities and {len(mine)} relations. Most connected entities: {', '.join(entities)}. Common relations: {', '.join(relations)}."
