"""
Tool Router package.

Exports:
    * :class:`RouterConfig` – configuration dataclass for the router.
    * :class:`ToolRouter` – the main orchestrator.
"""
from .config import RouterConfig
from .tool_router import ToolRouter

__all__ = ["RouterConfig", "ToolRouter"]
