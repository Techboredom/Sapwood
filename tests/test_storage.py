import json
from pathlib import Path

import pytest

from sapwood.storage import (
    BlockStore,
    SessionIndex,
    BlockNotFoundError,
    SessionNotFoundError,
    StorageError,
    block_id_for,
    model_slug,
)


MODEL = "test-model"


class TestBlockId:
    def test_stable_same_inputs(self):
        a = block_id_for(MODEL, 0, "tok_a", "prefix")
        b = block_id_for(MODEL, 0, "tok_a", "prefix")
        assert a == b

    def test_differs_by_position(self):
        assert block_id_for(MODEL, 0, "tok", "prefix") != block_id_for(MODEL, 16, "tok", "prefix")

    def test_differs_by_model(self):
        assert block_id_for("m1", 0, "tok", "prefix") != block_id_for("m2", 0, "tok", "prefix")

    def test_differs_by_kind(self):
        assert block_id_for(MODEL, 0, "tok", "prefix") != block_id_for(MODEL, 0, "tok", "generated")


class TestBlockStoreTokens:
    def test_store_persists_tokens_roundtrip(self, tmp_path):
        store = BlockStore(tmp_path)
        meta = store.store(
            block_id="toki", model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="h", token_count=4, payload=b"p", tokens=[7, 8, 9, 10],
        )
        assert meta["tokens"] == [7, 8, 9, 10]
        out_meta, _ = store.load("toki", model_id=MODEL)
        assert out_meta["tokens"] == [7, 8, 9, 10]

    def test_store_without_tokens_has_no_tokens_key(self, tmp_path):
        store = BlockStore(tmp_path)
        meta = store.store(
            block_id="notok", model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="h", token_count=4, payload=b"p",
        )
        assert "tokens" not in meta


class TestModelSlug:
    def test_slashes_become_underscores(self):
        assert "/" not in model_slug("org/model:v1")

    def test_empty_falls_back(self):
        assert model_slug("???") == "unknown"


class TestBlockStore:
    def test_store_load_roundtrip(self, tmp_path):
        store = BlockStore(tmp_path)
        payload = b"binary-kv-bytes" * 10
        meta = store.store(
            block_id="abc123", model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="tok", token_count=16, payload=payload,
        )
        assert meta["size_bytes"] == len(payload)

        out_meta, out_payload = store.load("abc123", model_id=MODEL)
        assert out_payload == payload
        assert out_meta["start_pos"] == 0
        assert out_meta["token_count"] == 16

    def test_exists(self, tmp_path):
        store = BlockStore(tmp_path)
        store.store(block_id="x", model_id=MODEL, start_pos=0, kind="prefix",
                    token_hash="h", token_count=16, payload=b"p")
        assert store.exists("x", model_id=MODEL)
        assert not store.exists("nope", model_id=MODEL)

    def test_load_missing_raises(self, tmp_path):
        store = BlockStore(tmp_path)
        with pytest.raises(BlockNotFoundError):
            store.load("nope", model_id=MODEL)

    def test_delete(self, tmp_path):
        store = BlockStore(tmp_path)
        store.store(block_id="to-del", model_id=MODEL, start_pos=0, kind="prefix",
                    token_hash="h", token_count=16, payload=b"p")
        assert store.delete("to-del", model_id=MODEL) is True
        assert store.exists("to-del", model_id=MODEL) is False

    def test_atomic_no_tmp_left(self, tmp_path):
        store = BlockStore(tmp_path)
        store.store(block_id="atomic", model_id=MODEL, start_pos=0, kind="prefix",
                    token_hash="h", token_count=16, payload=b"p")
        leftovers = list((tmp_path / "blocks").rglob("*.tmp.*"))
        assert leftovers == []


class TestSessionIndex:
    def test_create_get(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL, title="hello")
        assert s["model_id"] == MODEL
        assert s["state"] == "active"
        assert s["session_id"]

        fetched = idx.get(s["session_id"])
        assert fetched["session_id"] == s["session_id"]

    def test_create_duplicate_rejected(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL, session_id="fixed")
        with pytest.raises(StorageError):
            idx.create(MODEL, session_id="fixed")

    def test_append_block_idempotent(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL)
        desc = {"block_id": "b1", "start_pos": 0, "token_count": 16, "kind": "prefix"}
        idx.append_block(s["session_id"], desc)
        idx.append_block(s["session_id"], desc)  # duplicate
        fetched = idx.get(s["session_id"])
        assert len(fetched["blocks"]) == 1
        assert fetched["total_tokens"] == 16

    def test_list_filtered_by_model(self, tmp_path):
        idx = SessionIndex(tmp_path)
        idx.create(MODEL, title="a")
        idx.create("other-model", title="b")
        got = idx.list(model_id=MODEL)
        assert len(got) == 1
        assert got[0]["model_id"] == MODEL

    def test_update(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL)
        updated = idx.update(s["session_id"], state="paused", title="renamed")
        assert updated["state"] == "paused"
        assert updated["title"] == "renamed"

    def test_update_ignores_reserved_fields(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL)
        updated = idx.update(s["session_id"], blocks=[{"block_id": "x"}], created_at="bogus")
        assert updated["blocks"] == []
        assert updated["created_at"] == s["created_at"]

    def test_delete(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL)
        assert idx.delete(s["session_id"]) is True
        assert idx.delete(s["session_id"]) is False

    def test_missing_raises(self, tmp_path):
        idx = SessionIndex(tmp_path)
        with pytest.raises(SessionNotFoundError):
            idx.get("nonexistent")

    def test_append_block_persists_tokens_in_descriptor(self, tmp_path):
        idx = SessionIndex(tmp_path)
        s = idx.create(MODEL)
        desc = {"block_id": "b1", "start_pos": 0, "token_count": 4,
                "kind": "prefix", "tokens": [1, 2, 3, 4]}
        idx.append_block(s["session_id"], desc)
        fetched = idx.get(s["session_id"])
        b = fetched["blocks"][0]
        assert b["tokens"] == [1, 2, 3, 4]

    def test_path_traversal_guard(self, tmp_path):
        idx = SessionIndex(tmp_path)
        with pytest.raises(SessionNotFoundError):
            idx._path("../etc/passwd")
