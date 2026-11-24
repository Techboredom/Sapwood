"""Disk-backed, content-addressed KV block store + session index.

Storage model
-------------
A *block* is one KV-cache chunk for a single conversation, produced by vLLM.
Blocks are addressed by a stable logical identity (model + absolute start
position + hash of the block's token IDs + kind), not by the raw bytes, so the
same logical block computed in two different runs dedupes to one entry.

Layout under ``<root>/``::

    blocks/<model_slug>/<block_id[:2]>/<block_id>.bin
    blocks/<model_slug>/<block_id[:2]>/<block_id>.meta.json
    sessions/<session_id>.json

Writes are atomic (temp file + rename). Concurrency is expected to be handled by
the serving process (a single server owns the disk); this module is not itself
multi-process-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


class StorageError(Exception):
    """Base error for storage failures."""


class BlockNotFoundError(StorageError):
    pass


class SessionNotFoundError(StorageError):
    pass


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug for a model id."""
    slug = "".join(c if (c.isalnum() or c in "._-") else "_" for c in model_id)
    return slug.strip("._") or "unknown"


def validate_session_id(session_id: str) -> str:
    """Return a clean session id, or raise SessionNotFoundError.

    Shared by the disk and Valkey backends: only clean ids, no "..", no path
    separators — keeps session ids usable in key schemes and filenames alike.
    """
    import re
    safe = "".join(c for c in session_id if c.isalnum() or c in "._-")
    if not safe or ".." in safe or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", safe):
        raise SessionNotFoundError(f"invalid session id: {session_id!r}")
    return safe


def block_id_for(model_id: str, start_pos: int, token_hash: str, kind: str) -> str:
    """Stable logical identity for a KV block."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(int(start_pos)).encode("utf-8"))
    h.update(b"\x1f")
    h.update(token_hash.encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(kind).encode("utf-8"))
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{int(time.time()*1000)}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class BlockStore:
    """Stores and loads raw block bytes + their metadata sidecar."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "blocks").mkdir(parents=True, exist_ok=True)

    def _block_paths(self, model_id: str, block_id: str):
        slug = model_slug(model_id)
        shard = block_id[:2]
        folder = self.root / "blocks" / slug / shard
        return folder / f"{block_id}.bin", folder / f"{block_id}.meta.json"

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
        bin_path, meta_path = self._block_paths(model_id, block_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(bin_path, payload)
        _atomic_write(meta_path, json.dumps(meta, indent=2).encode("utf-8"))
        return meta

    def load(self, block_id: str, model_id: str | None = None) -> tuple[dict, bytes]:
        # Without a model id we can't know which shard to look in; require it.
        if model_id is None:
            # search all shards under blocks/
            for slug_dir in (self.root / "blocks").glob("*"):
                if not slug_dir.is_dir():
                    continue
                for shard_dir in slug_dir.glob("*"):
                    if not shard_dir.is_dir():
                        continue
                    bin_path = shard_dir / f"{block_id}.bin"
                    meta_path = shard_dir / f"{block_id}.meta.json"
                    if bin_path.is_file():
                        return json.loads(meta_path.read_text("utf-8")), bin_path.read_bytes()
            raise BlockNotFoundError(block_id)
        bin_path, meta_path = self._block_paths(model_id, block_id)
        if not bin_path.is_file():
            raise BlockNotFoundError(block_id)
        return json.loads(meta_path.read_text("utf-8")), bin_path.read_bytes()

    def exists(self, block_id: str, model_id: str | None = None) -> bool:
        try:
            self.load(block_id, model_id)
            return True
        except BlockNotFoundError:
            return False

    def delete(self, block_id: str, model_id: str | None = None) -> bool:
        try:
            bin_path, meta_path = self._block_paths(model_id, block_id) if model_id else (None, None)
            if model_id:
                removed = False
                for p in (bin_path, meta_path):
                    if p.is_file():
                        p.unlink(); removed = True
                return removed
            # search all shards
            removed = False
            for slug_dir in (self.root / "blocks").glob("*"):
                if not slug_dir.is_dir():
                    continue
                for shard_dir in slug_dir.glob("*"):
                    for suffix in (".bin", ".meta.json"):
                        p = shard_dir / f"{block_id}{suffix}"
                        if p.is_file():
                            p.unlink(); removed = True
            return removed
        except Exception:
            return False

    def size_bytes(self, block_id: str, model_id: str | None = None) -> int:
        meta, _ = self.load(block_id, model_id)
        return int(meta["size_bytes"])


class SessionIndex:
    """Persistent per-session index under ``<root>/sessions/``."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = validate_session_id(session_id)
        return self.sessions_dir / f"{safe}.json"

    def _read(self, session_id: str) -> dict:
        path = self._path(session_id)
        if not path.is_file():
            raise SessionNotFoundError(session_id)
        return json.loads(path.read_text("utf-8"))

    def _write(self, session: dict) -> None:
        session["updated_at"] = utc_now()
        _atomic_write(self._path(session["session_id"]),
                      json.dumps(session, indent=2).encode("utf-8"))

    def create(self, model_id: str, session_id: str | None = None,
               title: str | None = None) -> dict:
        import secrets
        session_id = session_id or f"sess_{secrets.token_hex(8)}"
        path = self._path(session_id)
        if path.is_file():
            raise StorageError(f"session already exists: {session_id}")
        session = {
            "session_id": session_id,
            "model_id": model_id,
            "title": title or session_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "state": "active",
            "total_tokens": 0,
            "blocks": [],
        }
        _atomic_write(path, json.dumps(session, indent=2).encode("utf-8"))
        return session

    def get(self, session_id: str) -> dict:
        return self._read(session_id)

    def list(self, model_id: str | None = None, state: str | None = None) -> list[dict]:
        out = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            try:
                s = json.loads(path.read_text("utf-8"))
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
        s = self._read(session_id)
        if any(b["block_id"] == block_descriptor["block_id"] for b in s["blocks"]):
            return s  # idempotent
        descriptor = {
            "block_id": block_descriptor["block_id"],
            "start_pos": int(block_descriptor["start_pos"]),
            "token_count": int(block_descriptor.get("token_count", 0)),
            "kind": str(block_descriptor.get("kind", "prefix")),
            "created_at": utc_now(),
        }
        # carry the token IDs so the retrieval layer can score relevance
        # without loading every block from disk
        toks = block_descriptor.get("tokens")
        if toks is not None:
            descriptor["tokens"] = [int(t) for t in toks]
        s["blocks"].append(descriptor)
        s["total_tokens"] = sum(b["token_count"] for b in s["blocks"])
        self._write(s)
        return s

    def update(self, session_id: str, **fields) -> dict:
        s = self._read(session_id)
        for k, v in fields.items():
            if k in ("session_id", "blocks", "created_at"):
                continue
            s[k] = v
        self._write(s)
        return s

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def summary(self, session_id: str) -> dict:
        s = self._read(session_id)
        total_blocks = len(s["blocks"])
        return {
            "session_id": session_id,
            "model_id": s["model_id"],
            "state": s["state"],
            "total_tokens": s["total_tokens"],
            "num_blocks": total_blocks,
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
        }
