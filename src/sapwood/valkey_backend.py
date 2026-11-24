"""Optional Valkey-backed storage backend (remote, network-shared).

The disk backend remains the default, fully-tested path. This module adds a
*second* backend that persists the same logical model — KV block blobs and the
per-session index — in a Valkey server instead of local files. Point vLLM's
connectors at one `sapwood-kv-server` and every instance sees the same store:
that is the "remote storage" option.

The backend talks to a Valkey server over a minimal, stdlib-only RESP2 client
(``valkey_client.py``) — no third-party dependency. Large values (payloads) are base64-wrapped into one JSON document per key so
a single ``SET``/``GET`` round-trips the block meta + payload atomically.
Keys:

* ``sb:blk:<model_slug>:<block_id>`` -> JSON ``{v, meta, payload_b64}``
* ``sb:sess:<session_id>`` -> JSON session descriptor

Session listing scans ``sb:sess:*`` and filters (simple, fine for small and
moderate session counts; not a design for 100k+ sessions).

Concurrency: single writer per session (same assumption as the disk store) —
``append_block`` is read-modify-write and not multi-process-atomic. ``create``
uses an atomic ``SET NX``.
"""

from __future__ import annotations

import base64
import json
import secrets

from .storage import (
    StorageError,
    BlockNotFoundError,
    SessionNotFoundError,
    model_slug,
    validate_session_id,
    utc_now,
)

KEY_PREFIX = "sb:"


def make_backend(config, client=None) -> tuple:
    """Return (ValkeyBlockStore, ValkeySessionIndex) for a ServerConfig.

    ``config.valkey_url`` is a URL like ``redis://host:port/db`` (the db index
    is ignored here — Valkey has one database for our use). An optional
    ``client`` can be injected for testing; production callers get a real one.
    """
    client = client if client is not None else _client_from_url(
        getattr(config, "valkey_url", "redis://127.0.0.1:6379/0"))
    return ValkeyBlockStore(client), ValkeySessionIndex(client)


def _client_from_url(url: str, timeout: float = 10.0):
    """Build a ValkeyClient from a redis:// URL."""
    from urllib.parse import urlsplit
    from .valkey_client import ValkeyClient
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 6379
    return ValkeyClient(host=host, port=port, timeout=timeout)


def _block_key(model_id: str, block_id: str) -> str:
    return f"{KEY_PREFIX}blk:{model_slug(model_id)}:{block_id}"


def _session_key(session_id: str) -> str:
    return f"{KEY_PREFIX}sess:{validate_session_id(session_id)}"


class ValkeyBlockStore:
    """Valkey-backed store with the same interface as storage.BlockStore."""

    def __init__(self, client):
        # client is an injected client (tests) — production callers pass a real one.
        self._client = client

    def _load_doc(self, raw) -> tuple[dict, bytes] | None:
        if raw is None:
            return None
        doc = json.loads(raw)
        payload = base64.b64decode(doc["payload"])
        return doc["meta"], payload

    def store(self, *, block_id: str, model_id: str, start_pos: int, kind: str,
              token_hash: str, token_count: int, payload: bytes,
              tokens: list | None = None) -> dict:
        meta = {
            "block_id": block_id,
            "model_id": model_id,
            "start_pos": int(start_pos),
            "kind": str(kind),
            "token_hash": str(token_hash),
            "token_count": int(token_count),
            "size_bytes": len(payload),
            "created_at": utc_now(),
        }
        if tokens is not None:
            meta["tokens"] = [int(t) for t in tokens]
        doc = {
            "v": 1,
            "meta": meta,
            "payload": base64.b64encode(payload).decode("ascii"),
        }
        self._client.set(_block_key(model_id, block_id), json.dumps(doc))
        return meta

    def load(self, block_id: str, model_id: str | None = None) -> tuple[dict, bytes]:
        if model_id is not None:
            raw = self._client.get(_block_key(model_id, block_id))
            loaded = self._load_doc(raw)
            if loaded is None:
                raise BlockNotFoundError(block_id)
            return loaded
        # search all model shards
        pattern = f"{KEY_PREFIX}blk:*:{block_id}"
        for raw_key in self._client.keys(pattern):
            raw = self._client.get(raw_key)
            loaded = self._load_doc(raw)
            if loaded is not None:
                return loaded
        raise BlockNotFoundError(block_id)

    def exists(self, block_id: str, model_id: str | None = None) -> bool:
        try:
            self.load(block_id, model_id)
            return True
        except BlockNotFoundError:
            return False

    def delete(self, block_id: str, model_id: str | None = None) -> bool:
        if model_id is not None:
            return bool(self._client.delete(_block_key(model_id, block_id)))
        pattern = f"{KEY_PREFIX}blk:*:{block_id}"
        deleted = False
        for key in self._client.keys(pattern):
            if self._client.delete(key):
                deleted = True
        return deleted

    def size_bytes(self, block_id: str, model_id: str | None = None) -> int:
        meta, _ = self.load(block_id, model_id)
        return int(meta["size_bytes"])


class ValkeySessionIndex:
    """Valkey-backed session index (same interface as storage.SessionIndex)."""

    def __init__(self, client):
        # client is an injected client (tests) — production callers pass a real one.
        self._client = client

    # -- helpers -----------------------------------------------------------
    def _get(self, session_id: str) -> dict | None:
        raw = self._client.get(_session_key(session_id))
        if raw is None:
            return None
        doc = json.loads(raw)
        return doc

    def _write(self, session: dict) -> None:
        session["updated_at"] = utc_now()
        self._client.set(_session_key(session["session_id"]), json.dumps(session))

    # -- CRUD --------------------------------------------------------------
    def create(self, model_id: str, session_id: str | None = None,
               title: str | None = None) -> dict:
        session_id = session_id or f"sess_{secrets.token_hex(8)}"
        safe = validate_session_id(session_id)
        key = _session_key(session_id)
        session = {
            "session_id": safe,
            "model_id": model_id,
            "title": title or safe,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "state": "active",
            "total_tokens": 0,
            "blocks": [],
        }
        written = self._client.set(key, json.dumps(session), nx=True)
        if not written:
            raise StorageError(f"session already exists: {safe}")
        return session

    def get(self, session_id: str) -> dict:
        session = self._get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def list(self, model_id: str | None = None, state: str | None = None) -> list[dict]:
        out = []
        for raw_key in self._client.keys(f"{KEY_PREFIX}sess:*"):
            raw = self._client.get(raw_key)
            try:
                s = json.loads(raw)
            except Exception:
                continue
            if model_id and s.get("model_id") != model_id:
                continue
            if state and s.get("state") != state:
                continue
            out.append(s)
        out.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return out

    def append_block(self, session_id: str, block_descriptor: dict) -> dict:
        session = self.get(session_id)  # raises if missing
        if any(b["block_id"] == block_descriptor["block_id"] for b in session["blocks"]):
            return session  # idempotent
        descriptor = {
            "block_id": block_descriptor["block_id"],
            "start_pos": int(block_descriptor["start_pos"]),
            "token_count": int(block_descriptor.get("token_count", 0)),
            "kind": str(block_descriptor.get("kind", "prefix")),
            "created_at": utc_now(),
        }
        toks = block_descriptor.get("tokens")
        if toks is not None:
            descriptor["tokens"] = [int(t) for t in toks]
        session["blocks"].append(descriptor)
        session["total_tokens"] = sum(b["token_count"] for b in session["blocks"])
        self._write(session)
        return session

    def update(self, session_id: str, **fields) -> dict:
        session = self.get(session_id)
        for k, v in fields.items():
            if k in ("session_id", "blocks", "created_at"):
                continue
            session[k] = v
        self._write(session)
        return session

    def delete(self, session_id: str) -> bool:
        return bool(self._client.delete(_session_key(session_id)))

    def summary(self, session_id: str) -> dict:
        s = self.get(session_id)
        return {
            "session_id": session_id,
            "model_id": s["model_id"],
            "state": s["state"],
            "total_tokens": s["total_tokens"],
            "num_blocks": len(s["blocks"]),
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
        }
