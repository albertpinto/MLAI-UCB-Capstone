"""The per-child LLM step: resolve references, then derive Q&A pairs, factoids and triplets from the resolved text.

Everything the LLM returns is checked before it is stored; anything that fails a check is dropped (or the child falls
back to its original text), because LLM output can be wrong.
"""
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .parsing import Parent

logger = logging.getLogger(__name__)

# Two calls, not one: when resolving references shares a prompt with the Q&A, facts and triplets, the model copies the text
# unchanged (measured: 12 pronouns before, 12 after). A prompt that only resolves, and lists its replacements first, works.
RESOLVE_SYSTEM = (
    "You resolve references inside one passage (CHILD) of a larger section (PARENT) so that the CHILD can be read alone. "
    "Find every pronoun (he, his, she, her, it, its, they, their, them) and vague reference (such person, the said Act, "
    "the former, this article, that House) in the CHILD, and for each give the exact words from the CHILD and the actual name it refers to, "
    "taken from the PARENT or the CHILD. Do not rewrite the CHILD; only list the replacements. "
    'Reply as JSON: {"replacements": [{"find": "his intention", "replace": "the President\'s intention"}]}'
)
DERIVE_SYSTEM = (
    "You prepare one passage for a search index. Reply as JSON with these keys: qa_pairs, factoids, triplets. "
    "qa_pairs: 2 to 4 pairs {question, answer} that a user would ask in everyday language, answered only from the passage. Every question "
    "must be self-contained: name its subject, and never say 'the text', 'this passage', 'the document' or 'the section'. "
    "factoids: 2 to 5 short standalone facts, one sentence each, each naming its subject (for example 'The President summons each House of "
    "Parliament.'). "
    "triplets: 2 to 6 facts as {subject, relation, object}: subject and object are short noun phrases copied from the passage (never "
    "pronouns), relation is a short verb phrase. "
    "Use only what the passage says; add nothing."
)

UNRESOLVED_OPENING = re.compile(r"^(this|that|these|those|it|they|he|she|such|the said|the former|the latter)\b", re.I)  # a fact must name its subject
VAGUE_QUESTION = re.compile(r"\b(the|this|that) (text|passage|document|section|excerpt)\b|mentioned in", re.I)
NUMBER = re.compile(r"\d+")
# Only a chunk with one of these can need resolving; skipping the rest saves a call (about 4 in 5 chunks come back unchanged).
REFERENCE = re.compile(
    r"\b(he|his|him|she|her|hers|it|its|they|their|them|such \w+|the said|the former|the latter"
    r"|(this|that|these|those) (article|clause|sub-clause|act|house|section|part|chapter|state|person|court|provision|schedule|right|law|authority|constitution))\b",
    re.I,
)
MAX_REPLACE_WORDS = 12
MAX_LENGTH_RATIO = 1.5  # resolving references adds words, but a rewrite much longer than the original has added content
MIN_GROUNDED = 0.6  # share of a fact's words that must appear in the text it was drawn from


@dataclass
class Enriched:
    content: str  # the child with references resolved (or the original if the rewrite failed a check)
    original: str
    qa_pairs: list[dict] = field(default_factory=list)
    factoids: list[str] = field(default_factory=list)
    triplets: list[dict] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 or w.isdigit()]


def grounded(fact: str, source: str) -> bool:
    """True if most of the fact's words occur in `source`: a cheap guard against invented facts."""
    words = _words(fact)
    if not words:
        return False
    present = set(_words(source))
    return sum(w in present for w in words) / len(words) >= MIN_GROUNDED


def resolved_ok(original: str, resolved: object) -> bool:
    """A rewrite is accepted only if it keeps every number and does not balloon in length."""
    if not isinstance(resolved, str) or not resolved.strip():
        return False
    ratio = len(resolved.split()) / max(len(original.split()), 1)
    return sorted(NUMBER.findall(original)) == sorted(NUMBER.findall(resolved)) and 0.7 <= ratio <= MAX_LENGTH_RATIO


def apply_replacements(child: str, raw: object) -> str:
    """Apply the LLM's {find, replace} list to `child` ourselves, so the text is never rewritten beyond those words."""
    text = child
    for r in sorted(raw if isinstance(raw, list) else [], key=lambda r: -len(str(r.get("find", ""))) if isinstance(r, dict) else 0):
        if not isinstance(r, dict) or not isinstance(r.get("find"), str) or not isinstance(r.get("replace"), str):
            continue
        find, replace = r["find"].strip(), r["replace"].strip()
        if find and replace and find != replace and len(replace.split()) <= MAX_REPLACE_WORDS:
            text = re.sub(rf"(?<!\w){re.escape(find)}(?!\w)", lambda _: replace, text)
    return text


def _clean_pairs(raw: object) -> list[dict]:
    pairs = raw if isinstance(raw, list) else []
    return [
        {"question": p["question"].strip(), "answer": p["answer"].strip()}
        for p in pairs
        if isinstance(p, dict) and isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)
        and p["question"].strip() and p["answer"].strip() and not VAGUE_QUESTION.search(p["question"])
    ]


def _clean_factoids(raw: object, source: str) -> list[str]:
    facts = raw if isinstance(raw, list) else []
    return [f.strip() for f in facts if isinstance(f, str) and 3 <= len(f.split()) <= 40 and not UNRESOLVED_OPENING.match(f.strip()) and grounded(f, source)]


def _clean_triplets(raw: object, source: str) -> list[dict]:
    triplets = raw if isinstance(raw, list) else []
    clean = []
    for t in triplets:
        if not isinstance(t, dict) or not all(isinstance(t.get(k), str) and t[k].strip() for k in ("subject", "relation", "object")):
            continue
        s, r, o = (t[k].strip() for k in ("subject", "relation", "object"))
        if len(r.split()) <= 6 and len(s.split()) <= 12 and len(o.split()) <= 12 and grounded(s, source) and grounded(o, source):
            clean.append({"subject": s, "relation": r, "object": o})
    return clean


class EnrichmentCache:
    """Append-only JSONL file of LLM results, so a crashed or repeated ingest does not pay for the same child twice."""

    def __init__(self, path: Path):
        self.path, self.lock, self.items = path, threading.Lock(), {}
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self.items[row["key"]] = row["reply"]
                except (json.JSONDecodeError, KeyError):
                    continue  # a half-written last line after a crash

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, reply: dict) -> None:
        with self.lock:
            self.items[key] = reply
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "reply": reply}) + "\n")


def cache_key(llm, system: str, user: str) -> str:
    """Changes whenever the prompt, the model or the text changes, so stale results are never reused."""
    return hashlib.sha256("\x00".join([system, str(getattr(llm, "model", "")), user]).encode()).hexdigest()


def _ask(llm, cache: EnrichmentCache | None, system: str, user: str) -> dict:
    """One cached LLM call; a failure gives {} so the other call and the other children carry on."""
    key = cache_key(llm, system, user)
    reply = cache.get(key) if cache else None
    if reply is None:
        try:
            reply = llm.json_call(system, user)
        except Exception as exc:
            logger.error("Chunk enrichment call failed: %s", exc)
            return {}
        if cache:
            cache.put(key, reply)
    return reply


def enrich_child(llm, settings: Settings, parent: Parent, child: str, cache: EnrichmentCache | None = None) -> Enriched:
    """Up to two LLM calls per child: resolve its references (only if it has any), then derive Q&A pairs, factoids and triplets from the resolved text.
    Never raises: whatever fails is left out, and the child keeps its original text."""
    try:
        section = f"SECTION: {parent.section or '-'}\n\n"
        content = child
        if REFERENCE.search(child):
            resolved = apply_replacements(child, _ask(llm, cache, RESOLVE_SYSTEM, f"{section}PARENT:\n{parent.text}\n\nCHILD:\n{child}").get("replacements"))
            content = resolved if resolved_ok(child, resolved) else child
        derived = _ask(llm, cache, DERIVE_SYSTEM, f"{section}PASSAGE:\n{content}")
        source = f"{content}\n{parent.text}"  # facts may use words from the surrounding section
        return Enriched(content, child, _clean_pairs(derived.get("qa_pairs")), _clean_factoids(derived.get("factoids"), source), _clean_triplets(derived.get("triplets"), source))
    except Exception as exc:  # one bad chunk must not abort the document
        logger.error("Chunk enrichment failed: %s", exc)
        return Enriched(child, child)
