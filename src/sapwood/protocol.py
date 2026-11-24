"""Length-prefixed framing over TCP.

Frame layout (all length prefixes big-endian uint32)::

    [4-byte header_len][header_len bytes][4-byte payload_len][payload]

`header` is a JSON object with a ``type`` field and any control fields.
`payload` is raw block bytes (or empty for control-only messages).

This keeps one connection able to carry both tiny control messages (JSON) and
multi-megabyte block transfers (binary) without any base64 overhead.
"""

from __future__ import annotations

import json
import socket

MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB guard


def read_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, or return None on EOF (clean or partial)."""
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> tuple[dict, bytes] | None:
    """Read one frame. Returns (header, payload) or None on clean EOF."""
    len_bytes = read_exactly(sock, 4)
    if len_bytes is None:
        return None
    header_len = int.from_bytes(len_bytes, "big")
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {header_len}")
    header_bytes = read_exactly(sock, header_len)
    if header_bytes is None:
        raise ConnectionError("connection closed mid-header")
    header = json.loads(header_bytes.decode("utf-8"))
    payload_len_bytes = read_exactly(sock, 4)
    if payload_len_bytes is None:
        raise ConnectionError("connection closed mid-payload-len")
    payload_len = int.from_bytes(payload_len_bytes, "big")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload too large: {payload_len}")
    payload = read_exactly(sock, payload_len) if payload_len else b""
    if payload is None:
        raise ConnectionError("connection closed mid-payload")
    return header, payload


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(h) > MAX_HEADER_BYTES:
        raise ValueError("header too large")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload too large")
    sock.sendall(len(h).to_bytes(4, "big") + h + len(payload).to_bytes(4, "big") + payload)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KVCacheClient:
    """Thin synchronous client for the cache server.

    One connection carries one request at a time (no pipelining); each call
    sends one frame and blocks for the reply. For a cache workload this is
    fine and keeps the protocol simple and debuggable.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7676, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "KVCacheClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure(self) -> socket.socket:
        if self._sock is None:
            self.connect()
        return self._sock  # type: ignore[return-value]

    def _request(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        sock = self._ensure()
        send_message(sock, header, payload)
        reply = recv_message(sock)
        if reply is None:
            raise ConnectionError("server closed the connection")
        resp_header, resp_payload = reply
        if resp_header.get("type") == "error":
            raise RuntimeError(resp_header.get("message", "server error"))
        return resp_header, resp_payload

    # control / lifecycle
    def ping(self) -> bool:
        resp, _ = self._request({"type": "ping"})
        return resp.get("type") == "pong"

    def create_session(self, model_id: str, session_id: str | None = None,
                       title: str | None = None) -> dict:
        h = {"type": "session_create", "model_id": model_id}
        if session_id:
            h["session_id"] = session_id
        if title:
            h["title"] = title
        resp, _ = self._request(h)
        return resp.get("session") or {}

    def get_session(self, session_id: str) -> dict:
        resp, _ = self._request({"type": "session_get", "session_id": session_id})
        return resp.get("session") or {}

    def list_sessions(self, model_id: str | None = None, state: str | None = None) -> list[dict]:
        h = {"type": "session_list"}
        if model_id:
            h["model_id"] = model_id
        if state:
            h["state"] = state
        resp, _ = self._request(h)
        return resp.get("sessions") or []

    def update_session(self, session_id: str, **fields) -> dict:
        h = {"type": "session_update", "session_id": session_id, **fields}
        resp, _ = self._request(h)
        return resp.get("session") or {}

    def delete_session(self, session_id: str) -> bool:
        resp, _ = self._request({"type": "session_delete", "session_id": session_id})
        return resp.get("status") == "ok"

    # blocks
    def exists_block(self, block_id: str, model_id: str | None = None) -> bool:
        h = {"type": "exists", "block_id": block_id}
        if model_id:
            h["model_id"] = model_id
        resp, _ = self._request(h)
        return bool(resp.get("found"))

    def store_block(self, *, block_id: str, model_id: str, start_pos: int, kind: str,
                    token_hash: str, token_count: int, session_id: str | None,
                    payload: bytes, tokens: list | None = None) -> dict:
        h = {
            "type": "store_block",
            "block_id": block_id,
            "model_id": model_id,
            "start_pos": int(start_pos),
            "kind": str(kind),
            "token_hash": str(token_hash),
            "token_count": int(token_count),
        }
        if tokens is not None:
            h["tokens"] = [int(t) for t in tokens]
        if session_id:
            h["session_id"] = session_id
        resp, _ = self._request(h, payload)
        return resp

    def load_block(self, block_id: str, model_id: str | None = None) -> tuple[dict, bytes]:
        h = {"type": "load_block", "block_id": block_id}
        if model_id:
            h["model_id"] = model_id
        resp, payload = self._request(h)
        found = bool(resp.get("found"))
        if not found:
            return {}, b""
        return resp, payload

    def delete_block(self, block_id: str, model_id: str | None = None) -> bool:
        h = {"type": "delete_block", "block_id": block_id}
        if model_id:
            h["model_id"] = model_id
        resp, _ = self._request(h)
        return resp.get("status") == "ok"
