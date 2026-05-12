"""
Embedding-based router backend.

Selects tools by ranking them according to semantic similarity with the query.
Supports five strategies:
    * ``local``      — sentence-transformers/all-MiniLM-L6-v2 (cached)
    * ``openai``     — OpenAI ``text-embedding-3-large`` via OpenAI-compatible API
    * ``qwen3``      — ``qwen3-embedding-8b`` via OpenAI-compatible API
    * ``bm25``       — BM25 keyword matching (no neural model)
    * ``contriever`` — ``facebook/contriever`` via remote embedding API
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcpuniverse.common.logger import get_logger
from mcpuniverse.tracer import Tracer
from mcpuniverse.agent.router.config import RouterConfig

from .base import RouterBackend

logger = get_logger("EmbeddingRouterBackend")


# ----------------------------------------------------------------------
# Strategy ABC
# ----------------------------------------------------------------------


class EmbeddingStrategy(ABC):
    """Compute similarity scores between a query and a list of tool texts."""

    @abstractmethod
    async def compute_scores(
        self,
        query: str,
        tool_texts: List[str],
        config: RouterConfig,
    ) -> List[float]:
        """
        Returns a list of floats with ``len(tool_texts)`` entries – higher is more relevant.
        """


# ----------------------------------------------------------------------
# Concrete strategies
# ----------------------------------------------------------------------


class LocalEmbeddingStrategy(EmbeddingStrategy):
    """Local sentence-transformers model (class-level cache)."""

    _model = None  # lazily loaded

    async def compute_scores(
        self, query: str, tool_texts: List[str], config: RouterConfig
    ) -> List[float]:
        import numpy as np  # local import to keep module import cheap

        if LocalEmbeddingStrategy._model is None:
            from sentence_transformers import SentenceTransformer

            LocalEmbeddingStrategy._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                local_files_only=True,
            )
        model = LocalEmbeddingStrategy._model
        tool_emb = np.array(
            model.encode(tool_texts, normalize_embeddings=True), dtype=np.float32
        )
        query_emb = np.array(
            model.encode(query, normalize_embeddings=True), dtype=np.float32
        )
        return np.dot(tool_emb, query_emb).tolist()


class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """OpenAI-compatible text-embedding-3-large."""

    _model_name = "text-embedding-3-large"

    async def compute_scores(
        self, query: str, tool_texts: List[str], config: RouterConfig
    ) -> List[float]:
        import httpx
        import numpy as np

        api_base = config.embedding_api_base or "https://api.openai.com/v1"
        api_key = config.embedding_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI embedding requires an API key. "
                "Set RouterConfig.embedding_api_key or the OPENAI_API_KEY env var."
            )

        all_texts = [query] + list(tool_texts)
        url = f"{api_base.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": self._model_name, "input": all_texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        embeddings = [item["embedding"] for item in result["data"]]
        query_emb = np.array(embeddings[0])
        tool_emb = np.array(embeddings[1:])
        query_norm = query_emb / np.linalg.norm(query_emb)
        tool_norms = tool_emb / np.linalg.norm(tool_emb, axis=1, keepdims=True)
        return np.dot(tool_norms, query_norm).tolist()


class Qwen3EmbeddingStrategy(OpenAIEmbeddingStrategy):
    """Qwen3 embedding via an OpenAI-compatible API."""

    _model_name = "qwen3-embedding-8b"

    async def compute_scores(
        self, query: str, tool_texts: List[str], config: RouterConfig
    ) -> List[float]:
        if not config.embedding_api_base:
            raise ValueError(
                "Qwen3 embedding requires RouterConfig.embedding_api_base to be set."
            )
        return await super().compute_scores(query, tool_texts, config)


class BM25Strategy(EmbeddingStrategy):
    """BM25 keyword search – no neural model, no API calls."""

    async def compute_scores(
        self, query: str, tool_texts: List[str], config: RouterConfig
    ) -> List[float]:
        from rank_bm25 import BM25Okapi

        corpus = [doc.lower().split() for doc in tool_texts]
        return BM25Okapi(corpus).get_scores(query.lower().split()).tolist()


class ContrieverStrategy(EmbeddingStrategy):
    """facebook/contriever via a remote embedding API."""

    async def compute_scores(
        self, query: str, tool_texts: List[str], config: RouterConfig
    ) -> List[float]:
        import httpx
        import numpy as np

        api_base = config.embedding_api_base
        if not api_base:
            raise ValueError(
                "Contriever embedding requires RouterConfig.embedding_api_base to be set."
            )
        api_key = config.embedding_api_key or ""

        all_texts = [query] + list(tool_texts)
        url = f"{api_base.rstrip('/')}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {"model": "facebook/contriever", "input": all_texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        embeddings = [item["embedding"] for item in result["data"]]
        query_emb = np.array(embeddings[0])
        tool_emb = np.array(embeddings[1:])
        # Server already L2-normalizes, so dot product == cosine similarity.
        return np.dot(tool_emb, query_emb).tolist()


# ----------------------------------------------------------------------
# Registry and backend
# ----------------------------------------------------------------------


EMBEDDING_STRATEGIES: Dict[str, type] = {
    "local": LocalEmbeddingStrategy,
    "openai": OpenAIEmbeddingStrategy,
    "qwen3": Qwen3EmbeddingStrategy,
    "bm25": BM25Strategy,
    "contriever": ContrieverStrategy,
}


class EmbeddingRouterBackend(RouterBackend):
    """Select tools by embedding similarity (or BM25)."""

    async def route(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        history: str,
        config: RouterConfig,
        tracer: Optional[Tracer] = None,
        callbacks: Optional[List] = None,
    ) -> List[str]:
        import numpy as np

        if not tools:
            return []

        # Build the search query (with optional history prepended).
        if config.embedding_use_history and history:
            search_query = f"{history}\n\nCurrent query: {query}"
        else:
            search_query = query

        tool_texts = [t.get("name", "") + " " + t.get("description", "") for t in tools]

        strategy_cls = EMBEDDING_STRATEGIES.get(config.embedding_model)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown embedding model: {config.embedding_model}. "
                f"Available: {list(EMBEDDING_STRATEGIES.keys())}"
            )
        scores = await strategy_cls().compute_scores(search_query, tool_texts, config)

        top_k = config.max_tools if config.max_tools > 0 else 5
        top_k = min(top_k, len(tools))
        top_indices = np.argsort(scores)[::-1][:top_k]

        selected = [tools[idx]["name"] for idx in top_indices]

        # Embedding calls don't go through BaseLLM, so record a tracer span manually.
        if tracer is not None:
            with tracer.sprout() as t:
                t.add({
                    "type": "router_embedding",
                    "embedding_model": config.embedding_model,
                    "query": query,
                    "available_tools_count": len(tools),
                    "selected_tool_names": selected,
                    "scores": [float(scores[int(i)]) for i in top_indices],
                })

        return selected
