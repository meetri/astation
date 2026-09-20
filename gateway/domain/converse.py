"""Context selection for `POST /api/converse` -- pure functions, no I/O."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")

_SPLIT_RE = re.compile(r"[_.\-/]+")

# Short on purpose: an aggressive stopword list starts deleting research vocabulary.
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
    """Lowercased tokens of `text`, compound tokens expanded into their parts."""
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
    """Distinct, order-preserving content terms: tokens minus stopwords."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(text):
        # Floor is 2, not 3: two-character terms (8b, s3, db, pr) carry real signal.
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


@dataclass
class ContextUnit:
    """One citable record from the gateway's own store."""

    kind: str
    ref: str
    label: str
    text: str
    timestamp: str | None = None
    detail: dict = field(default_factory=dict)
    primary: bool = False
    order: int = 0

    def terms(self) -> set[str]:
        return set(tokenize(self.text)) | set(tokenize(self.label))


@dataclass(frozen=True)
class SelectedUnit:
    """A unit that made it into the window, with why and how much of it did."""

    unit: ContextUnit
    number: int
    reason: str
    score: float
    matched_terms: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Selection:
    """The assembled window, plus everything needed to audit it."""

    units: tuple[SelectedUnit, ...]
    question_terms: tuple[str, ...]
    considered: int
    matched: int
    chars: int
    budget_chars: int
    truncated: bool


def idf_map(units: list[ContextUnit], terms: list[str]) -> dict[str, float]:
    """`term -> ln(1 + N / (1 + df))` over the candidate pool."""
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
    """`(score, matched terms)` -- IDF-weighted overlap, no term counted twice."""
    bag = unit.terms()
    matched = tuple(term for term in terms if term in bag)
    return sum(weights.get(term, 0.0) for term in matched), matched


def condense(text: str, terms: list[str], limit: int) -> str:
    """`text` shortened to about `limit` characters, keeping what was asked about."""
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
    """Choose the window: recency anchor first, then lexical relevance."""
    # A single 40 KB record must not take the whole window; cap it at a quarter.
    if per_unit_chars is None:
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

    # Anchors go in even at score zero: "what did we just do" has no lexical hook.
    for unit in anchors:
        score, matched = scores_by_unit[id(unit)]
        take(unit, "anchor", score, matched)

    by_kind: dict[str, list[tuple[ContextUnit, float, tuple[str, ...]]]] = {}
    for entry in sorted(scored, key=lambda e: (-e[1], -e[0].order)):
        # Never select a zero-score unit: padding the window is what invents answers.
        if entry[1] <= 0 or id(entry[0]) in anchor_refs:
            continue
        by_kind.setdefault(entry[0].kind, []).append(entry)
    rotation = sorted(by_kind, key=lambda kind: -by_kind[kind][0][1])
    cursors = dict.fromkeys(rotation, 0)
    # One unit per kind in rotation: a burst of same-kind rows must not bury the rest.
    while any(cursors[kind] < len(by_kind[kind]) for kind in rotation):
        for kind in rotation:
            index = cursors[kind]
            if index >= len(by_kind[kind]):
                continue
            cursors[kind] = index + 1
            unit, score, matched = by_kind[kind][index]
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
    """The numbered context block put in front of the question."""
    blocks: list[str] = []
    # The header carries no raw id: a model told to speak the answer would read it out.
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
