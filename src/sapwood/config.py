"""Configuration for the sapwood KV-cache stack."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 7676
    cache_dir: str = ".sapwood-cache"
    # "disk" (local, default) or "valkey" (remote, network-shared)
    backend: str = "disk"
    # used when backend == "valkey": redis://host:port/db
    valkey_url: str = "redis://127.0.0.1:6379/0"


@dataclass
class ClientConfig:
    host: str = "127.0.0.1"
    port: int = 7676
    timeout: float = 60.0
    # key used by the vLLM plugin to identify blocks
    model_id: str = "unknown"
    # how many hot blocks the GPU window keeps
    max_hot_blocks: int = 4
    # hard token budget for the hot window (None = unbounded, use max_hot_blocks)
    window_max_tokens: int | None = None
    # block size in tokens (must match the KV connector's block size)
    block_size: int = 16
    # pinned block ids kept in the window regardless of relevance (anchors / system prompt)
    pinned_blocks: list = field(default_factory=list)
    # weights for the lexical relevance scorer: (relevance, recency)
    relevance_weight: float = 1.0
    recency_weight: float = 0.25
    # fraction of window tokens allowed to be recomputed by the CacheBlend
    # quality-recovery path (0 disables recompute entirely)
    recompute_fraction: float = 0.15
