"""Minimal stdlib RESP2 client for a Valkey server.

Implements just what the sapwood Valkey backend needs — ``SET``, ``GET``,
``DEL``, ``EXISTS``, ``KEYS``, ``PING`` — over a plain TCP socket. Payload
bytes are base64-encoded by the caller (all values on the wire are ASCII), so
there is no binary-framing ambiguity.

This deliberately has *no* dependencies (no third-party client) so the
project stays stdlib-only. Responses: simple-string, error, integer, bulk
string, array. One persistent connection per instance; on a failed send/read
the connection is dropped and the next call reconnects.
"""

from __future__ import annotations

import socket


class ValkeyError(Exception):
    """Raised when the server replies with an error (or connection fails)."""


class ValkeyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, timeout: float = 10.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = bytearray()

    # -- connection lifecycle ────────────────────────────────────────────
    def _connect(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._buf = bytearray()

    def _close(self) -> None:
        sock, self._sock = self._sock, None
        self._buf = bytearray()
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def close(self) -> None:
        self._close()

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exc):
        self._close()
        return False

    # -- RESP codec ───────────────────────────────────────────────────────
    def _send(self, args: list[str]) -> None:
        self._connect()
        try:
            parts = [f"*{len(args)}\r\n".encode("utf-8")]
            for a in args:
                b = a.encode("utf-8") if isinstance(a, str) else a
                parts.append(f"${len(b)}\r\n".encode("utf-8"))
                parts.append(b)
                parts.append(b"\r\n")
            self._sock.sendall(b"".join(parts))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._close()
            raise ValkeyError(f"send failed: {e}") from e

    def _read_exact(self, n: int) -> bytes:
        assert self._sock is not None
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                self._close()
                raise ValkeyError("connection closed by server")
            self._buf += chunk
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def _readline(self) -> bytes:
        while True:
            idx = self._buf.find(b"\r\n")
            if idx >= 0:
                line = bytes(self._buf[:idx])
                del self._buf[: idx + 2]
                return line
            chunk = self._sock.recv(65536) if self._sock is not None else b""
            if not chunk:
                self._close()
                raise ValkeyError("connection closed by server")
            self._buf += chunk

    def _read_bulk(self, n: int) -> bytes:
        data = self._read_exact(n)
        tail = self._read_exact(2)
        if tail != b"\r\n":
            self._close()
            raise ValkeyError("malformed bulk string terminator")
        return data

    def _read_reply(self):
        try:
            head = self._readline()
        except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout) as e:
            self._close()
            raise ValkeyError(f"read failed: {e}") from e
        if not head:
            raise ValkeyError("empty reply")
        tag = head[0:1]
        rest = head[1:]
        if tag == b"+":
            return rest.decode("utf-8", "strict")
        if tag == b"-":
            raise ValkeyError(rest.decode("utf-8", "replace"))
        if tag == b":":
            return int(rest.decode("utf-8"))
        if tag == b"$":
            n = int(rest.decode("utf-8"))
            if n < 0:
                return None
            return self._read_bulk(n).decode("utf-8", "strict")
        if tag == b"*":
            n = int(rest.decode("utf-8"))
            if n < 0:
                return None
            return [self._read_reply() for _ in range(n)]
        raise ValkeyError(f"unknown reply type: {tag!r}")

    def _call(self, *args: str):
        self._send(list(args))
        return self._read_reply()

    # -- commands ─────────────────────────────────────────────────────────
    def ping(self) -> str:
        return self._call("PING")

    def set(self, key: str, value: str, nx: bool = False) -> bool:
        """SET key value [NX] — True if written, False if rejected by NX."""
        args = ["SET", key, value]
        if nx:
            args.append("NX")
        reply = self._call(*args)
        # RESP: +OK on success, $-1 (nil) when NX rejects
        if reply is None:
            return False
        return True

    def get(self, key: str) -> str | None:
        raw = self._call("GET", key)
        return raw  # None if missing

    def delete(self, *keys: str) -> int:
        return int(self._call("DEL", *keys))

    def exists(self, *keys: str) -> int:
        return int(self._call("EXISTS", *keys))

    def keys(self, pattern: str) -> list[str]:
        reply = self._call("KEYS", pattern)
        return [str(k) for k in (reply or [])]

    def scan(self, pattern: str, count: int = 1000) -> list[str]:
        """Safer than KEYS for large keyspaces: iterates cursor entries.

        Uses SCAN 0 with COUNT — suitable for small-to-moderate keysets here.
        """
        out: list[str] = []
        cursor = "0"
        while True:
            reply = self._call("SCAN", cursor, "MATCH", pattern, "COUNT", str(count))
            cursor = str(reply[0])
            out.extend(str(k) for k in reply[1])
            if cursor == "0":
                break
        return out
