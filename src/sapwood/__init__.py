"""sapwood-kv: network-based disk KV cache for vLLM.

Public API (stable, covered by tests):

* ``storage``  — BlockStore + SessionIndex (disk, content-addressable)
* ``protocol`` — framed TCP protocol (KVCacheClient, recv/send_message)
* ``server``   — CacheServer / CacheService (owns the disk store)
* ``retrieval``— BlockSelectionPolicy (RAG-style hot-window selection)
* ``valkey_backend`` — optional remote Valkey storage (stdlib-only RESP2 client)
* ``cacheblend`` — CacheBlend-style selective-recompute planning
* ``connector``— KVResumeManager (session resume without recompute)
* ``config``   — ServerConfig / ClientConfig
"""

from . import storage, protocol, retrieval, config, connector, server
from .storage import (
    BlockStore,
    SessionIndex,
    BlockNotFoundError,
    SessionNotFoundError,
    StorageError,
    block_id_for,
    model_slug,
)
from .protocol import KVCacheClient
from .server import CacheService, CacheServer
from .retrieval import (BlockSelectionPolicy, recency_scorer, assemble_window, window_summary, combined_score, token_jaccard)
from .cacheblend import compute_reuse_plan, reuse_plan_summary, ReusePlan
from .connector import KVResumeManager, build_default_manager
from .config import ServerConfig, ClientConfig
from .valkey_backend import make_backend as make_valkey_backend, ValkeyBlockStore, ValkeySessionIndex
from .valkey_client import ValkeyClient, ValkeyError
from . import vllm_adapter

__version__ = "0.1.0"

__all__ = [
    "BlockStore",
    "SessionIndex",
    "BlockNotFoundError",
    "SessionNotFoundError",
    "StorageError",
    "block_id_for",
    "model_slug",
    "KVCacheClient",
    "CacheService",
    "CacheServer",
    "BlockSelectionPolicy",
    "recency_scorer",
    "assemble_window",
    "window_summary",
    "combined_score",
    "token_jaccard",
    "compute_reuse_plan",
    "reuse_plan_summary",
    "ReusePlan",
    "KVResumeManager",
    "build_default_manager",
    "make_valkey_backend",
    "ValkeyBlockStore",
    "ValkeySessionIndex",
    "ValkeyClient",
    "ValkeyError",
    "ServerConfig",
    "ClientConfig",
    "vllm_adapter",
    "__version__",
]
