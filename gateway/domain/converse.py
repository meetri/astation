"""Context selection for `POST /api/converse` -- pure functions, no I/O.

**Context selection is the whole engineering problem here, and latency is
not.** Measured on this machine 2026-09-02: `llama3.2:3b` on Ollama answers in
about 2.8 seconds and every model tried refuses an unanswerable question in
under a second, so refusing is free. What is *not* free is answering from the
wrong slice: a benchmark of this exact task fed an 8B model a plain
"last N messages" window that did not contain the topic asked about, and it
produced a confident, well-formed, entirely fabricated conclusion -- while a
3B model on the same window correctly said the topic was not there. **A
confident liar with a good voice is worse than no feature at all.**

So this module exists to make "which records go in the window" a decision with
a name, rather than a slice off the end of a list.

## The strategy: anchor + lexical relevance, nothing cleverer

Every candidate record from the gateway's own store -- a transcript row, a run
event, a run, an artifact -- becomes one `ContextUnit` carrying the id it will
be CITED by. Then:

1. **Recency anchor.** The newest `anchor_units` units of the scope's PRIMARY
   narrative source are always included, whatever the question. Without this,
   "what did we just do" has no lexical hook and would select nothing.
2. **Lexical relevance, round-robin across kinds.** Every other unit is scored
   by IDF-weighted overlap with the question's content terms, computed over the
   candidate pool itself. Units are then taken one KIND at a time in rotation
   -- best transcript row, best run event, best artifact, next transcript
   row -- until the character budget is spent. Measured reason: on a real
   question a plain descending-score fill put thirteen near-identical artifact
   rows in the window and buried the one run event that answered it.
3. **Units that match nothing are never selected.** This is the property the
   whole design turns on. "Last N messages" pads the window with unrelated
   recent chatter, which is exactly the material a model mines to build a
   plausible-sounding answer about something that is not there. Here, a
   question about a topic the store has no record of produces a SHORT window
   containing only the anchor -- and a short window is the honest signal that
   there is nothing to answer from.
4. **Chronological in the prompt.** Selected units are presented oldest first
   and numbered, so the model reads a narrative rather than a relevance list,
   and so `SOURCES: 2, 5` maps back onto real ids.

IDF is computed over the candidate pool and not over some global corpus,
deliberately: the useful signal is "this term is rare *among the records that
could have answered this*", which is what makes a distinctive identifier
outrank a word that appears in every turn of the session.

**Why not embeddings, or FTS5, or an LLM re-ranker.** Each adds a component
that can be wrong in a way nobody can read off the response. This one can be
explained in a sentence and audited from `context.question_terms` and the
per-citation `matched_terms` the route returns: if the answer is bad, the
window that produced it is right there in the same JSON.

## Condensing one long unit

A single 40 KB tool result must not eat the whole budget. `condense()` keeps
the paragraphs of a unit that actually match the question, in their original
order, joined by an ellipsis -- and for a unit with no match at all (an anchor)
keeps the HEAD, because these texts lead with their outcome.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#: A token: a word, an identifier, a number, a path fragment. Applied to
#: already-lowercased text. Hyphens/dots/slashes/underscores are kept inside a
#: token so `run_events`, `20260829_223119_a8da0069` and `api/converse.py`
#: survive as single distinctive terms; `sub_tokens()` then also indexes their
#: parts, so asking about "converse" still matches "api/converse.py".
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")

#: Split a compound token into its parts.
_SPLIT_RE = re.compile(r"[_.\-/]+")

#: Words that carry no selection signal. Short and boring on purpose -- an
#: aggressive list starts deleting real research vocabulary. Everything here
#: is either English function-word noise or question furniture ("what", "did",
#: "tell", "me", "about") that appears in the question of every user.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cant",
        "could",
        "couldnt",
        "did",
        "didnt",
        "do",
        "does",
        "doesnt",
        "doing",
        "dont",
        "down",
        "during",
        "each",
        "even",
        "ever",
        "few",
        "for",
        "from",
        "further",
        "get",
        "got",
        "had",
        "hadnt",
        "has",
        "hasnt",
        "have",
        "havent",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isnt",
        "it",
        "its",
        "itself",
        "ive",
        "just",
        "know",
        "let",
        "like",
        "me",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "please",
        "same",
        "say",
        "see",
        "she",
        "should",
        "shouldnt",
        "so",
        "some",
        "such",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "us",
        "very",
        "was",
        "wasnt",
        "we",
        "were",
        "werent",
        "what",
        "whats",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "wont",
        "would",
        "wouldnt",
        "yes",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercased tokens of `text`, compound tokens expanded into their parts.

    `api/converse.py` yields `api/converse.py`, `api`, `converse` and `py`, so
    a question phrased either way matches. Never raises on odd input; a
    non-string is treated as empty.
    """
    if not isinstance(text, str) or not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("./-_")
        if not token:
            continue
        tokens.append(token)
        tokens.extend(part for part in _SPLIT_RE.split(token) if len(part) >= 3 and part != token)
    return tokens


def content_terms(text: str) -> list[str]:
    """Distinct, order-preserving content terms: tokens minus stopwords.

    Length 2 is the floor rather than 3 because this project's vocabulary is
    full of two-character terms that matter (`8b`, `s3`, `db`, `pr`).
    """
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(text):
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


@dataclass
class ContextUnit:
    """One citable record from the gateway's own store.

    `ref` is the id the answer will be attributed BY -- a transcript `row_id`,
    a `run_...` id, an `art_...` id -- and it is the reason this type exists
    instead of a plain string: a unit that cannot be cited has no business in
    the window.
    """

    #: `message` | `run_event` | `run` | `artifact` | `project` | `session`
    kind: str
    #: The citable id. Never a live Hermes handle.
    ref: str
    #: Short spoken-friendly description ("assistant reply", "artifact").
    label: str
    #: The record's text. Condensed at selection time, never here.
    text: str
    #: ISO-8601, or None for the (many) rows that carry no timestamp.
    timestamp: str | None = None
    #: Extra citation fields (`seq`, `source_path`, `role`, ...). Reported
    #: verbatim on the citation, so a claim can be traced to a byte on disk.
    detail: dict = field(default_factory=dict)
    #: Whether this unit belongs to the scope's PRIMARY narrative source --
    #: the only units the recency anchor may draw from.
    primary: bool = False
    #: Global append index, oldest first. Assigned by the assembler.
    order: int = 0

    def terms(self) -> set[str]:
        return set(tokenize(self.text)) | set(tokenize(self.label))


@dataclass(frozen=True)
class SelectedUnit:
    """A unit that made it into the window, with why and how much of it did."""

    unit: ContextUnit
    #: 1-based position in the prompt, which is what `SOURCES:` refers to.
    number: int
    #: `anchor` (newest primary records, always in) | `relevance`.
    reason: str
    score: float
    matched_terms: tuple[str, ...]
    #: The text actually placed in the prompt (possibly condensed).
    text: str


@dataclass(frozen=True)
class Selection:
    """The assembled window, plus everything needed to audit it."""

    units: tuple[SelectedUnit, ...]
    question_terms: tuple[str, ...]
    considered: int
    #: Candidates with a non-zero lexical score, before the budget was applied.
    matched: int
    chars: int
    budget_chars: int
    #: True when at least one scoring candidate was dropped for want of budget.
    truncated: bool


def idf_map(units: list[ContextUnit], terms: list[str]) -> dict[str, float]:
    """`term -> ln(1 + N / (1 + df))` over the candidate pool.

    Computed over the candidates rather than a global corpus on purpose: the
    question that matters is "is this term rare among the records that could
    have answered", so a term present in every turn of a session contributes
    almost nothing and a distinctive identifier dominates.
    """
    total = len(units)
    document_terms = [unit.terms() for unit in units]
    scores: dict[str, float] = {}
    for term in terms:
        frequency = sum(1 for bag in document_terms if term in bag)
        scores[term] = math.log(1 + total / (1 + frequency))
    return scores


def score_unit(
    unit: ContextUnit, terms: list[str], weights: dict[str, float]
) -> tuple[float, tuple[str, ...]]:
    """`(score, matched terms)` -- IDF-weighted overlap, no term counted twice.

    Presence, not frequency: a record that repeats a term forty times is not
    forty times more relevant, and counting frequency is how a long log file
    outranks the one sentence that answers the question.
    """
    bag = unit.terms()
    matched = tuple(term for term in terms if term in bag)
    return sum(weights.get(term, 0.0) for term in matched), matched


def condense(text: str, terms: list[str], limit: int) -> str:
    """`text` shortened to about `limit` characters, keeping what was asked about.

    Paragraphs that contain a question term are kept in their original order,
    joined with an ellipsis. A text with no match keeps its head -- these
    records lead with their outcome, so the head is the useful part when the
    unit is in the window as a recency anchor rather than on merit.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    wanted = set(terms)
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        blocks = [line for line in text.splitlines() if line.strip()]
    scored = [
        (index, block, len(wanted & set(tokenize(block)))) for index, block in enumerate(blocks)
    ]
    hits = [entry for entry in scored if entry[2] > 0]
    if not hits:
        return text[:limit].rstrip() + " ..."

    kept: list[tuple[int, str]] = []
    used = 0
    for index, block, _hits in sorted(hits, key=lambda e: (-e[2], e[0])):
        if used + len(block) > limit and kept:
            break
        piece = block if len(block) <= limit else block[:limit].rstrip() + " ..."
        kept.append((index, piece))
        used += len(piece)
        if used >= limit:
            break
    kept.sort()
    joined = " ... ".join(piece for _index, piece in kept)
    if kept[0][0] > 0:
        joined = "... " + joined
    if kept[-1][0] < len(blocks) - 1:
        joined = joined + " ..."
    return joined


def select_context(
    units: list[ContextUnit],
    question: str,
    *,
    budget_chars: int,
    anchor_units: int = 3,
    per_unit_chars: int | None = None,
) -> Selection:
    """Choose the window: recency anchor first, then lexical relevance.

    `units` arrive oldest-first with `order` already assigned. The anchor is
    the newest `anchor_units` PRIMARY units; everything else competes on
    IDF-weighted overlap and **a unit that matches no question term is never
    selected**, so an unanswerable question yields a short window instead of a
    padded one. Selected units come back chronological and numbered, which is
    the order they are printed in and the numbering `SOURCES:` refers to.
    """
    if per_unit_chars is None:
        # No single record may take more than a quarter of the window: one
        # 40 KB tool result would otherwise crowd out everything that
        # explains it.
        per_unit_chars = max(400, budget_chars // 4)

    terms = content_terms(question)
    weights = idf_map(units, terms)

    scored: list[tuple[ContextUnit, float, tuple[str, ...]]] = []
    for unit in units:
        score, matched = score_unit(unit, terms, weights)
        scored.append((unit, score, matched))
    matched_count = sum(1 for _u, score, _m in scored if score > 0)

    scores_by_unit = {id(unit): (score, matched) for unit, score, matched in scored}
    anchors = sorted(
        (unit for unit in units if unit.primary),
        key=lambda unit: unit.order,
        reverse=True,
    )[: max(0, anchor_units)]
    anchor_refs = {id(unit) for unit in anchors}

    chosen: dict[int, tuple[ContextUnit, str, float, tuple[str, ...], str]] = {}
    used = 0
    truncated = False

    def take(unit: ContextUnit, reason: str, score: float, matched: tuple[str, ...]) -> bool:
        nonlocal used, truncated
        if id(unit) in chosen:
            return True
        text = condense(unit.text, list(terms), per_unit_chars)
        if not text.strip():
            return True
        if used + len(text) > budget_chars and chosen:
            truncated = True
            return False
        chosen[id(unit)] = (unit, reason, score, matched, text)
        used += len(text)
        return True

    # 1. The anchor, newest first, so the freshest record survives a tight
    #    budget rather than being crowded out by an older one.
    for unit in anchors:
        score, matched = scores_by_unit[id(unit)]
        take(unit, "anchor", score, matched)

    # 2. Relevance, **round-robin across kinds**. Within a kind the order is
    #    descending score, ties breaking towards the newer record; across
    #    kinds one unit is taken at a time in rotation.
    #
    #    The rotation is not decoration -- it was added after a measured
    #    failure. Asked "why did CL28 fail?" against a real session, plain
    #    descending-score selection filled thirteen of nineteen slots with
    #    near-identical ARTIFACT rows whose titles all contain `cl28`
    #    (`cl28_zres.py`, `cl28_census.py`, ...), each scoring the same, and
    #    crowded out the one run event that actually states CL28's verdict.
    #    A burst of same-kind rows must not be able to bury the narrative:
    #    the top transcript row and the top run event now always precede the
    #    thirteenth file listing.
    by_kind: dict[str, list[tuple[ContextUnit, float, tuple[str, ...]]]] = {}
    for entry in sorted(scored, key=lambda e: (-e[1], -e[0].order)):
        if entry[1] <= 0 or id(entry[0]) in anchor_refs:
            continue
        by_kind.setdefault(entry[0].kind, []).append(entry)
    # Kinds enter the rotation in descending order of their best unit's score,
    # so the kind that matches the question best still goes first.
    rotation = sorted(by_kind, key=lambda kind: -by_kind[kind][0][1])
    cursors = dict.fromkeys(rotation, 0)
    while any(cursors[kind] < len(by_kind[kind]) for kind in rotation):
        for kind in rotation:
            index = cursors[kind]
            if index >= len(by_kind[kind]):
                continue
            cursors[kind] = index + 1
            unit, score, matched = by_kind[kind][index]
            # A unit that does not fit is skipped, not fatal: a later, smaller
            # one may still fit inside what is left.
            take(unit, "relevance", score, matched)

    ordered = sorted(chosen.values(), key=lambda entry: entry[0].order)
    selected = tuple(
        SelectedUnit(
            unit=unit,
            number=index,
            reason=reason,
            score=round(score, 4),
            matched_terms=matched,
            text=text,
        )
        for index, (unit, reason, score, matched, text) in enumerate(ordered, start=1)
    )
    return Selection(
        units=selected,
        question_terms=tuple(terms),
        considered=len(units),
        matched=matched_count,
        chars=sum(len(entry.text) for entry in selected),
        budget_chars=budget_chars,
        truncated=truncated,
    )


def render_context(selection: Selection) -> str:
    """The numbered context block put in front of the question.

    One record per entry, `[n]` then a one-line header naming what it is and
    when, then the text. The header carries the kind and timestamp but **never
    the raw id**: the id lives in the citation JSON, and putting a
    `run_3f9a1c2b...` in front of a model that has been told to speak its
    answer aloud invites it to read the id out.
    """
    blocks: list[str] = []
    for entry in selection.units:
        unit = entry.unit
        when = f", {unit.timestamp}" if unit.timestamp else ""
        blocks.append(f"[{entry.number}] {unit.label}{when}\n{entry.text}")
    return "\n\n".join(blocks)


__all__ = [
    "STOPWORDS",
    "ContextUnit",
    "SelectedUnit",
    "Selection",
    "condense",
    "content_terms",
    "idf_map",
    "render_context",
    "score_unit",
    "select_context",
    "tokenize",
]
