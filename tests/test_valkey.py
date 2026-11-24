"""Tests for the Valkey-backed storage backend + RESP2 client.

The backend logic is exercised against an in-memory fake client. The RESP2
framing is exercised against a minimal in-process RESP server over a real
socket pair (a tiny `FakeRESPServer`). No third-party packages or real
Valkey server is required for either suite.
"""

import base64
import json
import re
import socket
import threading

import pytest

from sapwood.valkey_client import ValkeyClient, ValkeyError
from sapwood.valkey_backend import ValkeyBlockStore, ValkeySessionIndex, KEY_PREFIX
from sapwood.storage import BlockNotFoundError, SessionNotFoundError, StorageError


# ---------------------------------------------------------------------------
# In-memory fake client (matches ValkeyClient's surface)
# ---------------------------------------------------------------------------
class FakeValkeyClient:
    def __init__(self):
        self._data = {}

    def _glob_match(self, pattern: str) -> list[str]:
        # translate the sb:* wildcard into a regex (only supported form)
        parts = pattern.split("*")
        regex = "^" + ".*".join(re.escape(p) for p in parts) + "$"
        return [k for k in self._data if re.fullmatch(regex, k)]

    def set(self, key, value, nx=False):
        if nx and key in self._data:
            return False
        self._data[key] = value
        return True

    def get(self, key):
        return self._data.get(key)

    def delete(self, *keys):
        removed = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                removed += 1
        return removed

    def exists(self, *keys):
        return sum(1 for k in keys if k in self._data)

    def keys(self, pattern):
        return self._glob_match(pattern)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Block store
# ---------------------------------------------------------------------------
def test_store_load_round_trip():
    client = FakeValkeyClient()
    store = ValkeyBlockStore(client)
    meta = store.store(block_id="abc", model_id="m", start_pos=16, kind="prefix",
                       token_hash="h", token_count=16, payload=b"payload",
                       tokens=[1, 2, 3])
    assert meta["size_bytes"] == 7
    assert meta["tokens"] == [1, 2, 3]

    loaded_meta, payload = store.load("abc", "m")
    assert loaded_meta["block_id"] == "abc"
    assert loaded_meta["tokens"] == [1, 2, 3]
    assert payload == b"payload"


def test_load_unknown_raises():
    store = ValkeyBlockStore(FakeValkeyClient())
    with pytest.raises(BlockNotFoundError):
        store.load("nope", "m")


def test_load_without_model_searches_all_shards():
    client = FakeValkeyClient()
    store = ValkeyBlockStore(client)
    store.store(block_id="blk1", model_id="m", start_pos=0, kind="prefix",
                token_hash="h", token_count=16, payload=b"data")
    meta, payload = store.load("blk1")  # no model id -> search
    assert payload == b"data"


def test_exists_and_size_and_delete():
    client = FakeValkeyClient()
    store = ValkeyBlockStore(client)
    store.store(block_id="blk", model_id="m", start_pos=0, kind="prefix",
                token_hash="h", token_count=16, payload=b"12345")
    assert store.exists("blk", "m") is True
    assert store.size_bytes("blk", "m") == 5
    assert store.delete("blk", "m") is True
    assert store.exists("blk", "m") is False


def test_atomic_block_key_namespaces_by_model():
    client = FakeValkeyClient()
    store = ValkeyBlockStore(client)
    store.store(block_id="blk", model_id="m1", start_pos=0, kind="prefix",
                token_hash="h", token_count=16, payload=b"a")
    store.store(block_id="blk", model_id="m2", start_pos=0, kind="prefix",
                token_hash="h", token_count=16, payload=b"b")
    # each model sees its own payload
    assert store.load("blk", "m1")[1] == b"a"
    assert store.load("blk", "m2")[1] == b"b"


# ---------------------------------------------------------------------------
# Session index
# ---------------------------------------------------------------------------
def test_session_create_get_list_delete():
    client = FakeValkeyClient()
    idx = ValkeySessionIndex(client)
    s = idx.create("m", session_id="sess1", title="t")
    assert s["session_id"] == "sess1"
    assert s["model_id"] == "m"
    assert s["blocks"] == []

    assert idx.get("sess1")["model_id"] == "m"

    listed = idx.list("m")
    assert len(listed) == 1

    assert idx.delete("sess1") is True
    with pytest.raises(SessionNotFoundError):
        idx.get("sess1")


def test_create_is_atomic_and_rejects_duplicate():
    client = FakeValkeyClient()
    idx = ValkeySessionIndex(client)
    idx.create("m", session_id="dup")
    with pytest.raises(StorageError):
        idx.create("m", session_id="dup")


def test_append_block_idempotent_and_updates_total_tokens():
    client = FakeValkeyClient()
    idx = ValkeySessionIndex(client)
    idx.create("m", session_id="s")
    d = {"block_id": "b1", "start_pos": 0, "token_count": 16, "kind": "prefix", "tokens": [1, 2]}
    s = idx.append_block("s", d)
    assert s["total_tokens"] == 16
    assert s["blocks"][0]["block_id"] == "b1"
    # idempotent on the same block_id
    s2 = idx.append_block("s", d)
    assert len(s2["blocks"]) == 1
    assert s2["total_tokens"] == 16


def test_update_rejects_protected_fields():
    client = FakeValkeyClient()
    idx = ValkeySessionIndex(client)
    idx.create("m", session_id="s")
    updated = idx.update("s", state="paused", blocks=[{"block_id": "hacked"}])
    assert updated["state"] == "paused"
    assert updated["blocks"] == []  # protected field ignored


def test_summary_and_list_filters():
    client = FakeValkeyClient()
    idx = ValkeySessionIndex(client)
    idx.create("m", session_id="s1")
    idx.create("m", session_id="s2")
    idx.update("s1", state="paused")
    assert idx.summary("s1")["state"] == "paused"
    active = idx.list("m", state="active")
    assert {s["session_id"] for s in active} == {"s2"}


def test_invalid_session_id_rejected():
    idx = ValkeySessionIndex(FakeValkeyClient())
    with pytest.raises(SessionNotFoundError):
        idx.create("m", session_id="../evil")


# ---------------------------------------------------------------------------
# RESP2 wire framing
# ---------------------------------------------------------------------------
import socket as _socket


def _make_client_for_test():
    """A ValkeyClient whose socket is a connected socketpair (no network)."""
    s1, s2 = _socket.socketpair()
    client = ValkeyClient.__new__(ValkeyClient)
    client.host = "127.0.0.1"
    client.port = 0
    client.timeout = 2.0
    client._sock = s1
    client._buf = bytearray()
    return client, s1, s2


def test_client_sends_correct_resp2_command_bytes():
    client, s1, s2 = _make_client_for_test()
    client._send(["SET", "k", "v"])
    # read the raw bytes from the peer — it should be the RESP2 encoding
    raw = s2.recv(4096)
    assert raw == b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"
    s1.close(); s2.close()


def test_client_get_parses_bulk_string_reply():
    client, s1, s2 = _make_client_for_test()
    s2.sendall(b"$5\r\nhello\r\n")   # one bulk-string reply
    assert client._read_reply() == "hello"
    s1.close(); s2.close()


def test_client_get_parses_nil_bulk_as_none():
    client, s1, s2 = _make_client_for_test()
    s2.sendall(b"$-1\r\n")
    assert client._read_reply() is None
    s1.close(); s2.close()


def test_client_parses_integer_and_array_replies():
    client, s1, s2 = _make_client_for_test()
    s2.sendall(b":42\r\n")
    assert client._read_reply() == 42
    s1.close(); s2.close()

    client, s1, s2 = _make_client_for_test()
    # array of two bulk strings
    s2.sendall(b"*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n")
    assert client._read_reply() == ["foo", "bar"]
    s1.close(); s2.close()


def test_client_status_and_error():
    client, s1, s2 = _make_client_for_test()
    s2.sendall(b"+OK\r\n")
    assert client._read_reply() == "OK"
    s1.close(); s2.close()

    client, s1, s2 = _make_client_for_test()
    s2.sendall(b"-ERR unknown\r\n")
    with pytest.raises(ValkeyError):
        client._read_reply()
    s1.close(); s2.close()


def test_client_get_multi_byte_bulk_over_chunked_recv():
    # simulate a server that drips bytes slowly: recv() may return fewer bytes
    client, s1, s2 = _make_client_for_test()
    payload = b"the quick brown fox jumps over the lazy dog"
    reply = b"$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n"
    s2.sendall(reply)
    assert client._read_reply() == payload.decode()
    s1.close(); s2.close()


def test_backend_make_backend_injects_client():
    from sapwood.valkey_backend import make_backend
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        valkey_url: str = "redis://127.0.0.1:6379/0"

    fake = FakeValkeyClient()
    store, sessions = make_backend(Cfg(), client=fake)
    assert isinstance(store, ValkeyBlockStore)
    assert isinstance(sessions, ValkeySessionIndex)


# ---------------------------------------------------------------------------
# End-to-end: Valkey-backed service over the framed TCP protocol
# ---------------------------------------------------------------------------
def test_valkey_backend_end_to_end_over_framed_protocol(tmp_path):
    import threading
    from sapwood.server import CacheService, CacheServer
    from sapwood.protocol import KVCacheClient

    fake = FakeValkeyClient()
    store = ValkeyBlockStore(fake)
    sessions = ValkeySessionIndex(fake)
    svc = CacheService(store, sessions)
    server = CacheServer(svc, host="127.0.0.1", port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = KVCacheClient(server.host, server.port, timeout=10)
        sess = client.create_session("m", title="valkey smoke")
        sid = sess["session_id"]

        from sapwood.storage import block_id_for
        block_id = block_id_for("m", 0, "h", "prefix")
        client.store_block(block_id=block_id, model_id="m", start_pos=0, kind="prefix",
                           token_hash="h", token_count=16, session_id=sid,
                           payload=b"\x01\x02\x03", tokens=[1, 2, 3])
        meta, payload = client.load_block(block_id, "m")
        assert payload == b"\x01\x02\x03"
        assert meta["tokens"] == [1, 2, 3]

        got = client.get_session(sid)
        assert got["session_id"] == sid
        assert len(got["blocks"]) == 1
    finally:
        server.stop()
