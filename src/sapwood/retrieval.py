"""RAG-style block selection / context assembly.

The disk store holds the whole conversation. GPU memory can hold only a hot
window. This module decides *which* blocks make it into that window each turn.

Two layers:

* ``BlockSelectionPolicy`` — the simple recency/pinned default (back-compatible
  with earlier call sites and tests).
* ``assemble_window`` — the full RAG-style path: given a query (and a budget),
  score every stored block by lexical relevance to the query plus a recency
  bonus, force-include pinned anchors, greedily fill the window budget, and
  return the selected blocks **in position order** so the model receives a
  coherent, ordered context.

Relevance is **lexical** (token-overlap) — a semantic embedding scorer can be
plugged in via the ``scorer`` argument. The disk store persists each block's
token IDs (``tokens`` in the meta sidecar) for this purpose.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# BlockSelectionPolicy (recency + pinned, back-compatible)
# ---------------------------------------------------------------------------

class BlockSelectionPolicy:
    """Decides the ordered list of block descriptors to load into GPU."""

    pinned: set[str]

    def __init__(self, max_blocks: int = 4, pinned: Iterable[str] | None = None,
                 scorer: Callable[[dict], float] | None = None):
        self.max_blocks = max(1, int(max_blocks))
        self.pinned = set(pinned or [])
        self.scorer = scorer

    def _pinned_descs(self, session: dict) -> list[dict]:
        return [b for b in session.get("blocks", []) if b["block_id"] in self.pinned]

    def _recency_descs(self, session: dict) -> list[dict]:
        return list(session.get("blocks", []))[-self.max_blocks:]

    def select(self, session: dict) -> list[dict]:
        if self.scorer is not None:
            return self._select_by_score(session)
        pinned = self._pinned_descs(session)
        recent = self._recency_descs(session)
        by_id: dict[str, dict] = {}
        for b in pinned:
            by_id[b["block_id"]] = b
        for b in recent:
            by_id.setdefault(b["block_id"], b)
        selected = list(by_id.values())
        selected.sort(key=lambda b: b.get("start_pos", 0))
        if len(selected) > self.max_blocks + len(self.pinned):
            pinned_ids = {b["block_id"] for b in pinned}
            recency_only = [b for b in selected if b["block_id"] not in pinned_ids]
            recency_only = recency_only[-self.max_blocks:]
            selected = sorted(pinned + recency_only, key=lambda b: b.get("start_pos", 0))
        return selected

    def _select_by_score(self, session: dict) -> list[dict]:
        blocks = list(session.get("blocks", []))
        scored = sorted(blocks, key=lambda b: self.scorer(b), reverse=True)
        chosen = scored[: self.max_blocks]
        pinned = self._pinned_descs(session)
        by_id: dict[str, dict] = {}
        for b in pinned:
            by_id[b["block_id"]] = b
        for b in chosen:
            by_id.setdefault(b["block_id"], b)
        selected = list(by_id.values())
        selected.sort(key=lambda b: b.get("start_pos", 0))
        return selected


def recency_scorer(anchor_pos: int = 0) -> Callable[[dict], float]:
    """A simple, deterministic default scorer: block value increases with
    absolute position (later blocks score higher)."""
    def scorer(block: dict) -> float:
        return float(block.get("start_pos", 0))
    return scorer


# ---------------------------------------------------------------------------
# Semantic / lexical scorer
# ---------------------------------------------------------------------------

def token_jaccard(block_tokens: Iterable[int], query_tokens: Iterable[int]) -> float:
    """Jaccard overlap between a block's token set and the query's token set."""
    a = set(block_tokens)
    b = set(query_tokens)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def block_tokens(block: dict) -> list[int]:
    """Tokens stored on a block descriptor (0/1 if unknown)."""
    return list(block.get("tokens") or [])


def token_overlap_score(block: dict, query_tokens: Iterable[int]) -> float:
    """Lexical relevance: Jaccard of the block's token set against the query."""
    return token_jaccard(block_tokens(block), query_tokens)


def recency_bonus(block: dict, ref_pos: int = 0) -> float:
    """Recency component, 0..1 — blocks closer to the tail of the conversation
    score higher. `ref_pos` is the conversation's effective tail (max pos)."""
    pos = block.get("start_pos", 0)
    if ref_pos <= 0:
        return 1.0 if pos >= 0 else 0.0
    return max(0.0, min(1.0, pos / ref_pos))


def combined_score(block: dict, query_tokens: Iterable[int], ref_pos: int,
                   weights: tuple[float, float] = (1.0, 0.25)) -> float:
    """relevance * w_rel + recency * w_rec."""
    relw, recw = weights
    return relw * token_overlap_score(block, query_tokens) + recw * recency_bonus(block, ref_pos)


# ---------------------------------------------------------------------------
# Window assembly
# ---------------------------------------------------------------------------

def assemble_window(
    session: dict,
    query_tokens: Optional[Iterable[int]] = None,
    *,
    max_blocks: int = 4,
    max_tokens: Optional[int] = None,
    pinned: Iterable[str] | None = None,
    anchor_blocks: Iterable[str] | None = None,
    weights: tuple[float, float] = (1.0, 0.25),
    scorer: Optional[Callable[[dict], float]] = None,
) -> list[dict]:
    """Assemble the hot window for a turn.

    Args:
        session:       the session record (with ``blocks`` list).
        query_tokens:  token IDs of the current query (for lexical scoring).
        max_blocks:    hard cap on number of blocks in the window.
        max_tokens:    hard cap on total tokens in the window (optional).
        pinned:        block ids that are always kept (anchors / system prompt).
        anchor_blocks: alias for pinned — system-prompt / anchor blocks.
        weights:       (relevance, recency) weights.
        scorer:        optional fully-custom scoring function ``block -> float``.

    Returns the selected blocks, ordered oldest-first (position order).
    """
    blocks = list(session.get("blocks", []))
    if not blocks:
        return []

    pinned_ids = set(pinned or []) | set(anchor_blocks or [])
    ref_pos = max((b.get("start_pos", 0) for b in blocks), default=0)

    def score_fn(b):
        if scorer is not None:
            return scorer(b)
        return combined_score(b, query_tokens or [], ref_pos, weights)

    # Force-include pinned anchors regardless of score
    forced = [b for b in blocks if b.get("block_id") in pinned_ids]
    rest = [b for b in blocks if b.get("block_id") not in pinned_ids]

    rest.sort(key=score_fn, reverse=True)

    selected: list[dict] = list(forced)
    selected_tokens = sum(b.get("token_count", 0) for b in forced)

    # Only non-pinned blocks count toward the max_blocks budget; pinned
    # anchors are over and above it (they are worth keeping regardless).
    non_pinned_slots = max_blocks

    for b in rest:
        if non_pinned_slots <= 0:
            break
        if max_tokens is not None and selected_tokens + b.get("token_count", 0) > max_tokens:
            continue  # skip, try the next (smaller/less relevant block)
        selected.append(b)
        selected_tokens += b.get("token_count", 0)
        non_pinned_slots -= 1

    # Return in conversation order (chronological)
    selected.sort(key=lambda b: b.get("start_pos", 0))
    return selected


def window_summary(blocks: list[dict]) -> dict:
    """Small summary of the assembled window (token count, span, blocks)."""
    total = sum(b.get("token_count", 0) for b in blocks)
    start = min((b.get("start_pos", 0) for b in blocks), default=0)
    end = max((b.get("start_pos", 0) + b.get("token_count", 0) for b in blocks), default=0)
    return {
        "num_blocks": len(blocks),
        "total_tokens": total,
        "span_start": start,
        "span_end": end,
        "span_tokens": end - start,
    }
