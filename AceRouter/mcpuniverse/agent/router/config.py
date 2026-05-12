"""
Configuration for the Tool Router.

Router is a standalone component used by agents (e.g., DynamicReAct).
LLM parameters for the router (model, temperature, seed, base_url, api_key, ...)
are delegated to the framework's LLM layer via ``RouterConfig.llm``.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from mcpuniverse.common.config import BaseConfig


@dataclass
class RouterConfig(BaseConfig):
    """
    Configuration for :class:`ToolRouter`.

    Attributes:
        mode: "llm" or "embedding".
        llm: LLM configuration dict for the router LLM (when ``mode == "llm"``).
            Format matches :class:`mcpuniverse.llm.manager.ModelManager.build_model`::

                {
                  "type": "openai",        # LLM type/alias (openai/claude/...)
                  "config": {              # Passed to the corresponding LLMConfig
                    "model_name": "router",
                    "base_url": "http://localhost:10121/v1",
                    "temperature": 1.0,
                    "seed": 42,
                    ...
                  }
                }

        embedding_model: One of "local", "openai", "qwen3", "bm25", "contriever"
            (when ``mode == "embedding"``).
        embedding_api_base: Base URL for remote embedding APIs.
        embedding_api_key: API key for remote embedding APIs.
        embedding_use_history: Concatenate history into the query for embedding search.
        max_tools: Maximum number of tools to return (``0`` means unlimited / default 5 for embedding).
        history_version: "v1" (execute-tool only) or "v2" (route + execute-tool).
            Default "v1" matches the training data format.
        enable_history: Whether to send history to the router.
        shuffle_tools: Randomly shuffle the tool list before sending to the router
            (position bias mitigation). Default False.
        system_prompt_template: Optional path to a custom router system-prompt Jinja2 template.
        user_prompt_template: Optional path to a custom router user-prompt Jinja2 template.
    """

    # --- Mode selection ---
    mode: str = "llm"

    # --- LLM backend ---
    llm: Dict = field(default_factory=dict)

    # --- Embedding backend ---
    embedding_model: str = "local"
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_use_history: bool = False

    # --- Tool selection ---
    max_tools: int = 0

    # --- History ---
    history_version: str = "v1"
    enable_history: bool = True

    # --- Tool presentation ---
    shuffle_tools: bool = False

    # --- Prompt template overrides (Jinja2 file paths) ---
    system_prompt_template: str = ""
    user_prompt_template: str = ""
