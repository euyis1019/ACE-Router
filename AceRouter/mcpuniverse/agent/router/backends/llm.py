"""
LLM-based router backend.

Reuses the framework's :class:`BaseLLM` + :class:`ModelManager` so that the router
LLM benefits from tracer integration, callbacks, retries, and unified
LLMConfig-based parameter management (``temperature``, ``seed``, ``max_tokens``,
``base_url``, ``api_key``, ...).

The router's *own* prompts (system / user) are rendered from Jinja2 templates
under ``mcpuniverse/agent/router/prompts/``.
"""
import json
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment

from mcpuniverse.common.logger import get_logger
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.tracer import Tracer
from mcpuniverse.agent.router.config import RouterConfig
from mcpuniverse.agent.router.response_parser import ResponseParser

from .base import RouterBackend

logger = get_logger("LLMRouterBackend")

# Prompt templates live next to the router package.
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "prompts"
)
DEFAULT_TEMPLATES = {
    "system_single": os.path.join(_PROMPTS_DIR, "router_system_single.j2"),
    "system_multi": os.path.join(_PROMPTS_DIR, "router_system_multi.j2"),
    "user_with_history": os.path.join(_PROMPTS_DIR, "router_user_with_history.j2"),
    "user_no_history": os.path.join(_PROMPTS_DIR, "router_user_no_history.j2"),
}


def _render_template(path: str, **kwargs) -> str:
    """Render a Jinja2 template file with the given variables."""
    with open(path, "r", encoding="utf-8") as fh:
        template_str = fh.read()
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template_str).render(**kwargs).strip()


class LLMRouterBackend(RouterBackend):
    """
    Selects tools by prompting an LLM built via :class:`ModelManager`.

    The router LLM is a *separate* instance from the reasoning LLM.  It has
    its own config (``RouterConfig.llm``) so that the two can differ in
    provider, model, temperature, seed, max_tokens, etc.
    """

    def __init__(self, config: RouterConfig):
        if not config.llm or "type" not in config.llm:
            raise ValueError(
                "RouterConfig.llm must specify 'type' and 'config'. "
                "Example: {'type': 'openai', 'config': {'model_name': 'router', ...}}"
            )
        manager = ModelManager()
        self._llm: BaseLLM = manager.build_model(
            name=config.llm["type"],
            config=config.llm.get("config", {}),
        )

    async def route(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        history: str,
        config: RouterConfig,
        tracer: Optional[Tracer] = None,
        callbacks: Optional[List] = None,
    ) -> List[str]:
        system_prompt = self._build_system_prompt(config)
        user_prompt = self._build_user_prompt(query, tools, history, config)

        raw = await self._llm.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tracer=tracer,
            callbacks=callbacks,
        )

        # Normalize to a plain string. Some BaseLLM implementations return a
        # provider-specific response object; extract the text content if so.
        if not isinstance(raw, str):
            if hasattr(raw, "choices"):
                raw = raw.choices[0].message.content  # type: ignore[attr-defined]
            else:
                raw = str(raw)

        return ResponseParser.parse_tool_names(raw)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, config: RouterConfig) -> str:
        if config.max_tools and config.max_tools > 1:
            path = config.system_prompt_template or DEFAULT_TEMPLATES["system_multi"]
            return _render_template(path, MAX_TOOLS=config.max_tools)
        path = config.system_prompt_template or DEFAULT_TEMPLATES["system_single"]
        return _render_template(path)

    def _build_user_prompt(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        history: str,
        config: RouterConfig,
    ) -> str:
        tools_json = json.dumps(tools)
        if config.enable_history and history:
            path = config.user_prompt_template or DEFAULT_TEMPLATES["user_with_history"]
            return _render_template(
                path, HISTORY=history, QUERY=query, TOOLS_JSON=tools_json
            )
        path = config.user_prompt_template or DEFAULT_TEMPLATES["user_no_history"]
        return _render_template(path, QUERY=query, TOOLS_JSON=tools_json)
