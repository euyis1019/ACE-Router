"""
ToolRouter — a standalone component for dynamic tool discovery.

Given a query and a list of tool descriptions, returns the names of the most
relevant tools.  It is framework-agnostic: it does not subclass ``BaseAgent``
and knows nothing about MCP servers.

Responsibilities:
    * Format the agent history via :class:`HistoryFormatter`
    * Optionally shuffle the tool list (position-bias mitigation)
    * Delegate to a :class:`RouterBackend` (LLM or embedding)
    * Return the selected tool *names* (name → full info resolution happens in the agent)
"""
import random
from typing import Any, Dict, List, Optional

from mcpuniverse.common.logger import get_logger
from mcpuniverse.tracer import Tracer

from .backends.base import RouterBackend
from .backends.embedding import EmbeddingRouterBackend
from .backends.llm import LLMRouterBackend
from .config import RouterConfig
from .history import HistoryFormatter

logger = get_logger("ToolRouter")


class ToolRouter:
    """
    Orchestrates history formatting, (optional) shuffling, and backend dispatch.
    """

    def __init__(self, config: RouterConfig):
        self._config = config
        self._backend: RouterBackend = self._create_backend()

    def _create_backend(self) -> RouterBackend:
        if self._config.mode == "llm":
            return LLMRouterBackend(self._config)
        if self._config.mode == "embedding":
            return EmbeddingRouterBackend()
        raise ValueError(f"Unknown router mode: {self._config.mode}")

    async def route(
        self,
        query: str,
        tools_for_routing: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        original_query: str = "",
        tracer: Optional[Tracer] = None,
        callbacks: Optional[List] = None,
    ) -> List[str]:
        """
        Args:
            query: The current routing query (usually the ``route`` action's ``arguments.query``).
            tools_for_routing: Prepared tool list
                ``[{"name": str, "description": str}, ...]`` — the agent is
                responsible for constructing this (flattening MCP tools, etc.).
            history: The agent's conversation history.
            original_query: The user's original message (used as the first-round
                user query when no explicit route query exists).
            tracer: Optional tracer (forwarded to the backend).
            callbacks: Optional callbacks (forwarded to the backend).

        Returns:
            A list of selected tool names.
        """
        # Position-bias mitigation.  Copy first so we don't mutate the caller's list.
        if self._config.shuffle_tools and len(tools_for_routing) > 1:
            tools_for_routing = list(tools_for_routing)
            random.shuffle(tools_for_routing)

        formatted_history = ""
        if self._config.enable_history and history:
            formatted_history = HistoryFormatter.format(
                history, original_query, self._config.history_version
            )

        return await self._backend.route(
            query=query,
            tools=tools_for_routing,
            history=formatted_history,
            config=self._config,
            tracer=tracer,
            callbacks=callbacks,
        )
