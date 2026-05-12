"""
Builds agent instances by type name or alias.
"""
# pylint: disable=too-few-public-methods
from typing import Dict, Optional, Union
from mcpuniverse.common.misc import BaseBuilder, ComponentABCMeta
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.base import BaseAgent


class AgentManager(BaseBuilder):
    """Factory for instantiating agents by class name or alias."""

    _AGENTS = ComponentABCMeta.get_class("agent")

    def __init__(self):
        super().__init__()
        self._classes = self._name_to_class(AgentManager._AGENTS)

    def build_agent(
        self,
        class_name: str,
        mcp_manager: MCPManager,
        config: Optional[Union[Dict, str]] = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Instantiate an agent by class name or alias.

        Args:
            class_name: Agent type name or alias (e.g. "react", "dynamic_react").
            mcp_manager: MCP server manager passed to the agent constructor.
            config: Agent configuration dict or path.
            **kwargs: Additional constructor arguments (e.g. ``llm``).
        """
        assert class_name in self._classes, (
            f"Agent '{class_name}' not found. Available: {list(self._classes.keys())}"
        )
        llm: BaseLLM = kwargs.pop("llm", None)
        return self._classes[class_name](mcp_manager=mcp_manager, llm=llm, config=config)
