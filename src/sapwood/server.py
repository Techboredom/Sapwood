"""Network cache server.

Owns the disk store and serves the framed protocol to any number of
clients (one thread per connection). Intended to be the single writer/reader
for a given disk directory — run it once, point the vLLM connectors at it, and
all instances share the same blocks and sessions.
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading

from .protocol import recv_message, send_message
from .storage import (
    BlockStore,
    SessionIndex,
    BlockNotFoundError,
    SessionNotFoundError,
    StorageError,
    block_id_for,
)
from .config import ServerConfig

log = logging.getLogger("sapwood.server")


class CacheService:
    """Request handler logic, decoupled from any transport so it is testable."""

    def __init__(self, store: BlockStore, sessions: SessionIndex):
        self.store = store
        self.sessions = sessions

    # -- blocks ------------------------------------------------------------
    def store_block(self, h: dict, payload: bytes) -> tuple[dict, bytes]:
        block_id = h.get("block_id")
        model_id = h.get("model_id")
        start_pos = h.get("start_pos")
        kind = h.get("kind", "prefix")
        token_hash = h.get("token_hash", "")
        token_count = int(h.get("token_count", 0))
        session_id = h.get("session_id")
        tokens = h.get("tokens")
        if not block_id:
            return {"type": "stored", "status": "error", "message": "missing block_id"}, b""
        # recompute the id from the logical identity to catch caller errors
        expected = block_id_for(model_id, start_pos, token_hash, kind)
        if expected != block_id:
            return {
                "type": "stored",
                "status": "error",
                "message": f"block_id mismatch: expected {expected}, got {block_id}",
            }, b""
        meta = self.store.store(
            block_id=block_id, model_id=model_id, start_pos=int(start_pos),
            kind=kind, token_hash=token_hash, token_count=token_count, payload=payload,
            tokens=tokens if tokens is not None else None,
        )
        if session_id:
            try:
                d = {
                    "block_id": block_id,
                    "start_pos": int(start_pos),
                    "token_count": token_count,
                    "kind": kind,
                }
                if tokens is not None:
                    d["tokens"] = tokens
                self.sessions.append_block(session_id, d)
            except SessionNotFoundError:
                pass  # block stored even if the session is unknown/missing
        return {"type": "stored", "status": "ok", "block_id": block_id, **meta}, b""

    def load_block(self, h: dict) -> tuple[dict, bytes]:
        block_id = h.get("block_id")
        model_id = h.get("model_id")
        try:
            meta, payload = self.store.load(block_id, model_id)
            resp = {"type": "loaded", "found": True, "block_id": block_id,
                    "model_id": meta["model_id"], "start_pos": meta["start_pos"],
                    "kind": meta["kind"], "token_count": meta["token_count"],
                    "size_bytes": meta["size_bytes"]}
            if "tokens" in meta:
                resp["tokens"] = meta["tokens"]
            return resp, payload
        except BlockNotFoundError:
            return {"type": "loaded", "found": False, "block_id": block_id}, b""

    def exists_block(self, h: dict) -> tuple[dict, bytes]:
        block_id = h.get("block_id")
        model_id = h.get("model_id")
        return {"type": "exists_resp", "block_id": block_id,
                "found": self.store.exists(block_id, model_id)}, b""

    def delete_block(self, h: dict) -> tuple[dict, bytes]:
        block_id = h.get("block_id")
        model_id = h.get("model_id")
        removed = self.store.delete(block_id, model_id)
        return {"type": "deleted", "block_id": block_id,
                "status": "ok" if removed else "not_found"}, b""

    # -- sessions ----------------------------------------------------------
    def create_session(self, h: dict) -> tuple[dict, bytes]:
        try:
            session = self.sessions.create(
                model_id=h.get("model_id", "unknown"),
                session_id=h.get("session_id"),
                title=h.get("title"),
            )
            return {"type": "session_created", "session": session}, b""
        except StorageError as e:
            return {"type": "session_created", "status": "error", "message": str(e)}, b""

    def get_session(self, h: dict) -> tuple[dict, bytes]:
        try:
            s = self.sessions.get(h.get("session_id", ""))
            return {"type": "session_found", "session": s}, b""
        except SessionNotFoundError:
            return {"type": "session_found", "status": "not_found"}, b""

    def list_sessions(self, h: dict) -> tuple[dict, bytes]:
        model_id = h.get("model_id")
        state = h.get("state")
        return {"type": "sessions_listed",
                "sessions": self.sessions.list(model_id=model_id, state=state)}, b""

    def update_session(self, h: dict) -> tuple[dict, bytes]:
        session_id = h.get("session_id", "")
        fields = {k: v for k, v in h.items()
                  if k not in ("type", "session_id")}
        try:
            s = self.sessions.update(session_id, **fields)
            return {"type": "session_updated", "session": s}, b""
        except SessionNotFoundError:
            return {"type": "session_updated", "status": "not_found"}, b""

    def delete_session(self, h: dict) -> tuple[dict, bytes]:
        removed = self.sessions.delete(h.get("session_id", ""))
        return {"type": "session_deleted", "status": "ok" if removed else "not_found"}, b""

    # -- entry point -------------------------------------------------------
    def dispatch(self, header: dict, payload: bytes) -> tuple[dict, bytes]:
        msg_type = header.get("type")
        try:
            if msg_type == "ping":
                return {"type": "pong"}, b""
            if msg_type == "store_block":
                return self.store_block(header, payload)
            if msg_type == "load_block":
                return self.load_block(header)
            if msg_type == "exists":
                return self.exists_block(header)
            if msg_type == "delete_block":
                return self.delete_block(header)
            if msg_type == "session_create":
                return self.create_session(header)
            if msg_type == "session_get":
                return self.get_session(header)
            if msg_type == "session_list":
                return self.list_sessions(header)
            if msg_type == "session_update":
                return self.update_session(header)
            if msg_type == "session_delete":
                return self.delete_session(header)
            return {"type": "error", "message": f"unknown message type: {msg_type}"}, b""
        except StorageError as e:
            return {"type": "error", "message": str(e)}, b""
        except Exception as e:  # don't leak a stack; keep the connection alive
            log.exception("handler error")
            return {"type": "error", "message": f"handler error: {e}"}, b""


class CacheServer:
    """Threaded TCP server; one thread per client connection."""

    def __init__(self, service: CacheService, host: str = "127.0.0.1", port: int = 7676):
        self.service = service
        self.host = host
        self.requested_port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(128)
        self.port = self._sock.getsockname()[1]
        self._running = True
        self._threads: list[threading.Thread] = []
        self._connections = 0
        self._conns_lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port

    def serve_forever(self) -> None:
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle_conn, args=(conn, addr),
                                 name=f"client-{addr[0]}:{addr[1]}", daemon=True)
            t.start()
            with self._conns_lock:
                self._connections += 1
                self._threads.append(t)

    def _handle_conn(self, conn: socket.socket, addr) -> None:
        try:
            with conn:
                conn.settimeout(120.0)
                while self._running:
                    reply = recv_message(conn)
                    if reply is None:
                        break
                    header, payload = reply
                    resp_header, resp_payload = self.service.dispatch(header, payload)
                    send_message(conn, resp_header, resp_payload)
        except (socket.timeout, ConnectionError, ValueError):
            pass
        except Exception:
            log.exception("connection handler error")
        finally:
            with self._conns_lock:
                self._connections -= 1

    def stop(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass
        for t in list(self._threads):
            t.join(timeout=2.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sapwood KV-cache server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7676)
    parser.add_argument("--cache-dir", required=False,
                        help="directory for the disk store (disk backend; required for --backend=disk)")
    parser.add_argument("--backend", choices=("disk", "valkey"), default="disk",
                        help="storage backend (default: disk)")
    parser.add_argument("--valkey-url", default="redis://127.0.0.1:6379/0",
                        help="Valkey URL when --backend=valkey")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    config = ServerConfig(
        host=args.host, port=args.port, cache_dir=args.cache_dir,
        backend=args.backend, valkey_url=args.valkey_url,
    )
    if config.backend == "valkey":
        from .valkey_backend import make_backend
        store, sessions = make_backend(config)
        backend_desc = f"valkey={args.valkey_url}"
    else:
        if not config.cache_dir:
            parser.error("--cache-dir is required for --backend=disk (default)")
        store = BlockStore(config.cache_dir)
        sessions = SessionIndex(config.cache_dir)
        backend_desc = f"disk={config.cache_dir}"

    service = CacheService(store, sessions)
    server = CacheServer(service, host=config.host, port=config.port)
    log.info("listening on %s:%d (%s)", server.host, server.port, backend_desc)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
