"""Client-side bridge between a vLLM request and the disk cache.

Transport-agnostic and dependency-free: works against the framed TCP protocol
(``protocol.py``) and the disk store (``storage.py``). Fully testable without
a GPU or a model.

The vLLM plugin itself is a thin integration layer documented in
``vllm_adapter.py``; it is not available here and is not verified.
"""

from __future__ import annotations

from typing import Optional

from .config import ClientConfig
from .protocol import KVCacheClient
from .storage import block_id_for
from .retrieval import (
    BlockSelectionPolicy,
    assemble_window,
    window_summary,
)
from .cacheblend import compute_reuse_plan, ReusePlan


class KVResumeManager:
    """Wraps a KVCacheClient to do session-resume + retrieval-assembly."""

    def __init__(self, client: KVCacheClient, config: Optional[ClientConfig] = None):
        self.client = client
        self.config = config or ClientConfig()

    # -- session lifecycle ────────────────────────────────────────────────
    def ensure_session(self, session_id: str | None, model_id: str, title: str | None = None) -> dict:
        if session_id:
            try:
                return self.client.get_session(session_id)
            except RuntimeError:
                pass
        return self.client.create_session(model_id=model_id, session_id=session_id, title=title)

    def delete_session(self, session_id: str) -> bool:
        return self.client.delete_session(session_id)

    # -- block storage ───────────────────────────────────────────────────
    def store_block(self, *, session_id: str, model_id: str, start_pos: int,
                    token_hash: str, token_count: int, kind: str, payload: bytes,
                    tokens: list | None = None) -> dict:
        block_id = block_id_for(model_id, start_pos, token_hash, kind)
        return self.client.store_block(
            block_id=block_id, model_id=model_id, start_pos=start_pos, kind=kind,
            token_hash=token_hash, token_count=token_count, session_id=session_id,
            payload=payload, tokens=tokens,
        )

    def load_block(self, block_id: str, model_id: str | None = None) -> tuple[dict, bytes]:
        return self.client.load_block(block_id, model_id)

    # -- retrieval-assembly ─────────────────────────────────────────────
    def assemble_context(self, session_id: str, query_tokens: list | None = None, *,
                         max_blocks: int | None = None, max_tokens: int | None = None,
                         pinned: list | None = None) -> tuple[list[dict], dict]:
        """Select the hot window for a turn using the RAG-style policy.

        Returns ``(selected_blocks, summary)`` where ``selected_blocks`` is the
        ordered list of block descriptors (oldest first) that fit the budget.
        The caller then loads their payloads via :meth:`resume_session`.

        ``max_blocks`` / ``max_tokens`` default to the config's window budget.
        """
        session = self.client.get_session(session_id)
        if not session:
            return [], {}

        max_blocks = max_blocks if max_blocks is not None else self.config.max_hot_blocks
        max_tokens = max_tokens if max_tokens is not None else self.config.window_max_tokens
        pinned = pinned if pinned is not None else self.config.pinned_blocks

        selected = assemble_window(
            session, query_tokens,
            max_blocks=max_blocks, max_tokens=max_tokens,
            pinned=pinned, anchor_blocks=pinned,
            weights=(self.config.relevance_weight, self.config.recency_weight),
        )
        summary = window_summary(selected)
        return selected, summary


    # -- CacheBlend-style quality recovery ───────────────────────────────
    def assemble_window_with_reuse(self, session_id: str, query_tokens: list | None = None,
                                   *, max_blocks: int | None = None,
                                   max_tokens: int | None = None,
                                   pinned: list | None = None,
                                   recompute_fraction: float | None = None) -> tuple[list[dict], dict, ReusePlan]:
        """Assemble the hot window *and* decide which blocks to recompute.

        This is the CacheBlend quality-recovery pattern over the assembled window:
        given the ordered selection from ``assemble_context``, decide which blocks
        are stale (their cached KV was computed against a prefix that was skipped)
        — recompute those, reuse the rest — within ``recompute_fraction`` of the
        window's tokens.

        Returns ``(selected_blocks, summary, reuse_plan)``. Callers then
        recompute the recompute blocks (model-side) and load the reusable blocks
        via :meth:`load_block` / :meth:`resume_session`.
        """
        session = self.client.get_session(session_id)
        if not session:
            return [], {}, ReusePlan()

        selected, summary = self.assemble_context(
            session_id, query_tokens, max_blocks=max_blocks, max_tokens=max_tokens, pinned=pinned,
        )
        pinned = pinned if pinned is not None else self.config.pinned_blocks
        frac = recompute_fraction if recompute_fraction is not None else self.config.recompute_fraction
        plan = compute_reuse_plan(selected, recompute_fraction=frac, pinned=pinned)
        return selected, summary, plan

    def recompute_blocks(self, session_id: str, query_tokens: list | None = None, *,
                         max_blocks: int | None = None, max_tokens: int | None = None,
                         pinned: list | None = None,
                         recompute_fraction: float | None = None) -> list[dict]:
        """Blocks the CacheBlend path says must be recomputed this turn."""
        _, _, plan = self.assemble_window_with_reuse(
            session_id, query_tokens, max_blocks=max_blocks, max_tokens=max_tokens,
            pinned=pinned, recompute_fraction=recompute_fraction,
        )
        return plan.recompute

    # -- resume ──────────────────────────────────────────────────────────
    def resume_session(self, session_id: str,
                       policy: BlockSelectionPolicy | None = None,
                       *,
                       query_tokens: list | None = None,
                       max_blocks: int | None = None,
                       max_tokens: int | None = None,
                       pinned: list | None = None) -> list[tuple[dict, bytes]]:
        """Return the blocks selected for THIS turn, loaded from disk, in order.

        This is the whole point of the cache: a resumed session reuses stored
        blocks instead of re-running the prefill. Each returned entry is
        ``(meta, payload)`` oldest-first.

        If ``policy`` is given it is used exactly as before (back-compat).
        Otherwise the RAG-style :func:`assemble_window` path is used, with the
        optional ``query_tokens``, ``max_blocks`` / ``max_tokens`` budget, and
        ``pinned`` anchors.
        """
        session = self.client.get_session(session_id)
        if not session:
            return []

        if policy is not None:
            selected = policy.select(session)
        else:
            selected, _ = self.assemble_context(
                session_id, query_tokens,
                max_blocks=max_blocks, max_tokens=max_tokens, pinned=pinned,
            )

        out: list[tuple[dict, bytes]] = []
        for b in selected:
            try:
                meta, payload = self.client.load_block(b["block_id"], session.get("model_id"))
            except RuntimeError:
                continue
            if payload:
                out.append((meta, payload))
        return out

    def missing_blocks(self, session_id: str,
                       policy: BlockSelectionPolicy | None = None,
                       query_tokens: list | None = None) -> list[dict]:
        """Blocks the window wants that are not present on disk (need recompute)."""
        session = self.client.get_session(session_id)
        if not session:
            return []
        if policy is not None:
            selected = policy.select(session)
        else:
            selected, _ = self.assemble_context(session_id, query_tokens)
        missing = []
        for b in selected:
            found = self._safe_exists(b["block_id"], session.get("model_id"))
            if not found:
                missing.append(b)
        return missing

    def _safe_exists(self, block_id: str, model_id: str | None) -> bool:
        try:
            return bool(self.client.exists_block(block_id, model_id))
        except RuntimeError:
            return False

    def session_summary(self, session_id: str) -> dict:
        try:
            session = self.client.get_session(session_id)
        except RuntimeError:
            return {}
        return {
            "session_id": session.get("session_id"),
            "model_id": session.get("model_id"),
            "state": session.get("state"),
            "total_tokens": session.get("total_tokens"),
            "num_blocks": len(session.get("blocks", [])),
            "updated_at": session.get("updated_at"),
        }


def build_default_manager(host: str, port: int, model_id: str,
                          max_hot_blocks: int = 4, pinned_blocks: list | None = None,
                          timeout: float = 60.0) -> KVResumeManager:
    """Convenience factory."""
    client = KVCacheClient(host, port, timeout=timeout)
    config = ClientConfig(model_id=model_id, max_hot_blocks=max_hot_blocks,
                          pinned_blocks=pinned_blocks or [])
    return KVResumeManager(client, config)
