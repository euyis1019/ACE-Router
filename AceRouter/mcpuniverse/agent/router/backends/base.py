"""Abstract base class for Router backends (LLM / embedding)."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcpuniverse.tracer import Tracer
from mcpuniverse.agent.router.config import RouterConfig


class RouterBackend(ABC):
    """A pluggable backend that selects relevant tools for a query."""

    @abstractmethod
    async def route(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        history: str,
        config: RouterConfig,
        tracer: Optional[Tracer] = None,
        callbacks: Optional[List] = None,
    ) -> List[str]:
        """
        Select the most relevant tools for ``query``.

        Args:
            query: The query text.
            tools: Candidate tools as ``[{"name": str, "description": str}, ...]``.
            history: Pre-formatted history string (empty when disabled).
            config: Router configuration.
            tracer: Optional tracer (forwarded to the underlying LLM when applicable).
            callbacks: Optional callbacks (forwarded to the underlying LLM when applicable).

        Returns:
            A list of selected tool names (preserves order, may be empty).
        """
