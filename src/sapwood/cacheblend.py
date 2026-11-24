"""CacheBlend-style selective recompute for a replayed window.

When an assembled window (selected by ``retrieval.assemble_window``) is
replayed into the model, a block's cached KV can be *stale* relative to what
the model actually sees: the KV for a block was computed under a prefix that
included the blocks we skipped. The window therefore isn't a perfect
contiguous prefix replay, and reused blocks near the gaps are inaccurate.

The CacheBlend pattern (non-prefix reuse with quality recovery) addresses
this by recomputing only the most-affected tokens and reusing the rest.

We cannot compute hidden states here — this layer has no model and no GPU.
So the staleness signal is a *structural proxy* derived from the window
itself:

* a kept block is stale if the block immediately before it in the window does
  **not** end exactly where it starts (a gap = skipped prefix context that the
  block's hidden states were computed against);
* the first kept block is the assumed base of the replayed context (staleness 0);
* pinned anchors (system prompt / ground truth) are trusted and never
  recomputed — staleness 0 by definition.

We then recompute the most-stale blocks until the recompute-token budget is
exhausted; the remaining blocks are marked reusable. This is block-granular
(the atomic unit), not token-granular as in the original CacheBlend paper —
that would require the model's attention statistics, which are out of scope
for this layer. The seam for an engine-level scorer is ``staleness_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class ReusePlan:
    """Structured decision: which blocks to recompute, which to reuse as-is."""

    reusable: list[dict] = field(default_factory=list)   # full block dicts, start_pos order
    recompute: list[dict] = field(default_factory=list)  # full block dicts, start_pos order
    staleness: dict = field(default_factory=dict)        # block_id -> gap tokens (0 if fresh)

    @property
    def total_window_tokens(self) -> int:
        return sum(b.get("token_count", 0) for b in self.reusable + self.recompute)

    @property
    def recompute_tokens(self) -> int:
        return sum(b.get("token_count", 0) for b in self.recompute)

    @property
    def recompute_fraction(self) -> float:
        total = self.total_window_tokens
        return (self.recompute_tokens / total) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "reusable_ids": [b["block_id"] for b in self.reusable],
            "recompute_ids": [b["block_id"] for b in self.recompute],
            "reusable": [dict(b) for b in self.reusable],
            "recompute": [dict(b) for b in self.recompute],
            "staleness": dict(self.staleness),
            "total_window_tokens": self.total_window_tokens,
            "recompute_tokens": self.recompute_tokens,
            "recompute_fraction": self.recompute_fraction,
        }


def compute_reuse_plan(
    selected: list[dict],
    *,
    recompute_fraction: float = 0.15,
    pinned: Iterable[str] | None = None,
    staleness_fn: Callable[[dict, dict | None], float] | None = None,
) -> ReusePlan:
    """Decide which blocks to recompute vs reuse as-is.

    Args:
        selected:        the selected window (ordered or unordered; sorted by start_pos).
        recompute_fraction:
            budget as a fraction of the window's total tokens. The most-stale
            blocks are recomputed until this budget is exhausted. Guaranteed to
            recompute at least the single most-stale block when any are stale
            (a block is the atomic unit — we cannot sub-select tokens without
            model statistics). Set to 0 to disable recompute entirely.
        pinned:          block ids trusted as-is (never recomputed).
        staleness_fn:    optional fully-custom ``(block, prev_block_or_none) -> float``
                        staleness scorer. Default: gap-token proxy.

    Returns:
        A :class:`ReusePlan` (ordered oldest-first within each bucket).
    """
    ordered = sorted(selected, key=lambda b: b.get("start_pos", 0))
    pinned_ids = set(pinned or [])

    # --- staleness per block ---
    staleness: dict[str, float] = {}
    for i, b in enumerate(ordered):
        if b["block_id"] in pinned_ids:
            staleness[b["block_id"]] = 0.0
            continue
        if i == 0:
            # first kept block = assumed base of the replayed context
            staleness[b["block_id"]] = 0.0
            continue
        prev = ordered[i - 1]
        if staleness_fn is not None:
            staleness[b["block_id"]] = float(staleness_fn(b, prev))
            continue
        gap = b.get("start_pos", 0) - (prev.get("start_pos", 0) + prev.get("token_count", 0))
        staleness[b["block_id"]] = float(max(0, gap))

    # --- select blocks to recompute, most-stale first, within the budget ---
    total_window_tokens = sum(b.get("token_count", 0) for b in ordered)
    budget = int(recompute_fraction * total_window_tokens)

    stale = [b for b in ordered if staleness[b["block_id"]] > 0]
    stale.sort(key=lambda b: staleness[b["block_id"]], reverse=True)

    recompute: list[dict] = []
    recompute_ids: set[str] = set()
    recompute_tokens = 0
    if recompute_fraction > 0:
        for b in stale:
            blk_tokens = b.get("token_count", 0)
            if recompute_tokens == 0 or recompute_tokens + blk_tokens <= budget:
                recompute.append(b)
                recompute_ids.add(b["block_id"])
                recompute_tokens += blk_tokens
            else:
                break

    recompute.sort(key=lambda b: b.get("start_pos", 0))
    reusable = [b for b in ordered if b["block_id"] not in recompute_ids]

    return ReusePlan(
        reusable=reusable,
        recompute=recompute,
        staleness=staleness,
    )


def reuse_plan_summary(plan: ReusePlan) -> dict:
    """Small, log-friendly summary of a :class:`ReusePlan`."""
    return {
        "num_reusable": len(plan.reusable),
        "num_recompute": len(plan.recompute),
        "recompute_tokens": plan.recompute_tokens,
        "total_window_tokens": plan.total_window_tokens,
        "recompute_fraction": plan.recompute_fraction,
        "recompute_ids": [b["block_id"] for b in plan.recompute],
        "reusable_ids": [b["block_id"] for b in plan.reusable],
    }
