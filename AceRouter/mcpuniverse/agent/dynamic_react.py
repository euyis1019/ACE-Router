"""
A dynamic tool-discovery ReAct agent.

Follows the :class:`ReAct` pattern:
    * Inherits :class:`BaseAgent` (framework-managed LLM, MCP tools, tracer, callbacks).
    * Inlines the reasoning loop inside ``_execute``.
    * Uses ``self._tools`` (loaded by ``BaseAgent.initialize``) and ``self.call_tool``.

Difference from :class:`ReAct`:
    * Owns a :class:`ToolRouter` component (``self._router``).
    * The reasoning LLM emits JSON with two extra action types:
        - ``route``:       delegates tool discovery to ``self._router``.
        - ``execute-tool``: executes a concrete MCP tool discovered by ``route``.
"""
# pylint: disable=broad-exception-caught
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from mcp.types import TextContent

from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.router import RouterConfig, ToolRouter
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.agent.utils import build_system_prompt
from mcpuniverse.callbacks.base import (
    CallbackMessage,
    MessageType,
    send_message,
    send_message_async,
)
from mcpuniverse.common.logger import get_logger
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.tracer import Tracer

DEFAULT_CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

PROMPT_VERSIONS: Dict[str, str] = {
    "v1": os.path.join(DEFAULT_CONFIG_FOLDER, "dynamic_react_prompt.j2"),
    "react": os.path.join(DEFAULT_CONFIG_FOLDER, "react_prompt.j2"),
}

# Tool results longer than this are truncated before being sent back to the LLM.
MAX_TOOL_RESULT_LENGTH = 500_000


@dataclass
class DynamicReActConfig(BaseAgentConfig):
    """
    Configuration for :class:`DynamicReAct`.

    Mirrors :class:`ReActConfig`: only contains agent-level logic parameters.
    Tool management (which servers, tool whitelists, permissions) lives in
    ``BaseAgentConfig.servers``; reasoning-LLM parameters live in the framework's
    LLM config; router LLM parameters live in ``router.llm``.
    """

    system_prompt: str = ""
    prompt_version: str = "v1"
    context_examples: str = ""
    max_iterations: int = 5
    summarize_tool_response: bool = False
    router: RouterConfig = field(default_factory=RouterConfig)


class DynamicReAct(BaseAgent):
    """
    Dynamic tool-discovery ReAct agent.

    The LLM emits JSON with ``thought`` and either ``action`` or ``answer``::

        {"thought": "...", "action": {"tool": "route", "arguments": {"query": "..."}}}
        {"thought": "...", "action": {"tool": "execute-tool",
                                       "arguments": {"server_name": "...", "tool_name": "...", "params": {...}}}}
        {"thought": "...", "answer": "..."}
    """

    config_class = DynamicReActConfig
    alias = ["dynamic_react", "dynamic_mcp", "dynamic"]

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._logger = get_logger(f"{self.__class__.__name__}:{self._name}")
        self._history: List[Dict[str, str]] = []
        self._router = ToolRouter(self._config.router)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str) -> str:
        params: Dict[str, Any] = {
            "INSTRUCTION": self._config.instruction,
            "QUESTION": question,
            "MAX_STEPS": self._config.max_iterations,
        }
        if self._config.context_examples:
            params["CONTEXT_EXAMPLES"] = self._config.context_examples
        params.update(self._config.template_vars)

        if self._config.system_prompt:
            prompt_template = self._config.system_prompt
        else:
            version = self._config.prompt_version
            if version not in PROMPT_VERSIONS:
                self._logger.warning(
                    "Unknown prompt_version '%s', falling back to 'v1'", version
                )
                version = "v1"
            prompt_template = PROMPT_VERSIONS[version]

        return build_system_prompt(
            system_prompt_template=prompt_template,
            tool_prompt_template=self._config.tools_prompt,
            tools={},
            **params,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _add_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Return the agent's conversation history."""
        return self._history

    def clear_history(self) -> None:
        """Clear the agent's conversation history."""
        self._history = []

    def reset(self):
        """Reset the agent between task runs."""
        self.clear_history()

    # ------------------------------------------------------------------
    # Tool preparation – build inputs for the router from ``self._tools``
    # ------------------------------------------------------------------

    def _prepare_tools_for_routing(self):
        """
        Build ``(tools_for_routing, tools_full_info)`` from ``self._tools``.

        ``self._tools`` is populated by :meth:`BaseAgent.initialize` from
        ``self._config.servers`` and maps ``server_name -> [mcp.types.Tool, ...]``.
        """
        tools_for_routing: List[Dict[str, Any]] = []
        tools_full_info: Dict[str, Dict[str, Any]] = {}
        for server_name, tools in self._tools.items():
            for tool in tools:
                tools_for_routing.append({
                    "name": tool.name,
                    "description": tool.description,
                })
                tools_full_info[tool.name] = {
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema,
                }
        return tools_for_routing, tools_full_info

    @staticmethod
    def _format_route_result(
        selected_names: List[str], tools_full_info: Dict[str, Dict[str, Any]]
    ) -> str:
        """Render the router's selection into a string for the reasoning LLM.

        Tool names that don't exist in ``tools_full_info`` (hallucinated by the
        router) are silently dropped here; they simply don't appear in the
        reasoning LLM's view, so the LLM will choose something else next turn.
        """
        real = [tools_full_info[n] for n in selected_names if n in tools_full_info]
        lines = [f"Found {len(real)} relevant tools:\n"]
        for info in real:
            lines.append(f"- Server: {info['server']}, Tool: {info['name']}")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  inputSchema: {json.dumps(info['schema'], indent=2)}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON extraction (robust to think-tags, code fences, minor format errors)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_from_response(response: str) -> Optional[Dict]:
        if not response:
            return None
        text = response.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in [r"\{.*\}", r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
            for match in re.findall(pattern, text, re.DOTALL):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        try:
            fixed = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
            fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)
            fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
            fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
            fixed = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", fixed)
            return json.loads(re.sub(r"\s+", " ", fixed).strip())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Main reasoning loop
    # ------------------------------------------------------------------

    async def _execute(
        self,
        message: Union[str, List[str]],
        output_format: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> AgentResponse:
        if isinstance(message, (list, tuple)):
            message = "\n".join(message)
        if output_format is not None:
            message = message + "\n\n" + self._get_output_format_prompt(output_format)

        tracer: Tracer = kwargs.get("tracer", Tracer())
        callbacks = kwargs.get("callbacks", [])

        # Prepared once at the start of this execution; tools don't change mid-loop.
        tools_for_routing, tools_full_info = self._prepare_tools_for_routing()

        system_prompt = self._build_prompt(message)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please help me complete this task step by step."},
        ]
        messages.extend(self._history)

        for iter_num in range(self._config.max_iterations):
            # --- 1. Reasoning LLM call (tracer + callbacks are auto-recorded by BaseLLM) ---
            response = await self._llm.generate_async(
                messages=messages, tracer=tracer, callbacks=callbacks
            )
            if not isinstance(response, str):
                response = str(response)

            self._add_history("assistant", response)
            messages.append({"role": "assistant", "content": response})

            parsed = self._extract_json_from_response(response)
            if not parsed or "thought" not in parsed:
                err = "Invalid response format"
                self._logger.warning(err)
                self._add_history("user", f"Error: {err}")
                messages.append({"role": "user", "content": f"Error: {err}"})
                continue

            # --- 2. Final answer ---
            if "answer" in parsed and parsed["answer"] is not None:
                answer = parsed["answer"]
                if not isinstance(answer, str):
                    answer = json.dumps(answer, ensure_ascii=False)
                await self._send_callback_message(
                    callbacks, iter_num, thought=parsed["thought"], answer=answer
                )
                return AgentResponse(
                    name=self._name,
                    class_name=self.__class__.__name__,
                    response=answer,
                    trace_id=tracer.trace_id,
                )

            # --- 3. Action dispatch ---
            action = parsed.get("action") or {}
            tool_name = action.get("tool", "") if isinstance(action, dict) else ""

            if tool_name == "route":
                query = action.get("arguments", {}).get("query", message)
                try:
                    selected_names = await self._router.route(
                        query=query,
                        tools_for_routing=tools_for_routing,
                        history=self.get_history(),
                        original_query=message,
                        tracer=tracer,
                        callbacks=callbacks,
                    )
                    result_text = self._format_route_result(selected_names, tools_full_info)
                except Exception as exc:
                    self._logger.error("Route failed: %s", exc)
                    result_text = f"Error: Route failed: {str(exc)[:300]}"
                self._add_history("user", result_text)
                messages.append({"role": "user", "content": result_text})
                await self._send_callback_message(
                    callbacks, iter_num,
                    thought=parsed["thought"], action=action, result=result_text,
                )

            elif tool_name == "execute-tool":
                args = action.get("arguments", {}) or {}
                tool_action = {
                    "server": args.get("server_name"),
                    "tool": args.get("tool_name"),
                    "arguments": args.get("params", {}) or {},
                }
                try:
                    tool_result = await self.call_tool(
                        tool_action, tracer=tracer, callbacks=callbacks
                    )
                    tool_content = tool_result.content[0]
                    if not isinstance(tool_content, TextContent):
                        raise ValueError("Tool output is not text")
                    result_text = tool_content.text
                    if self._config.summarize_tool_response:
                        result_text = await self.summarize_tool_response(
                            result_text,
                            context=json.dumps(tool_action, indent=2),
                            tracer=tracer,
                        )
                    if len(result_text) > MAX_TOOL_RESULT_LENGTH:
                        result_text = (
                            result_text[:MAX_TOOL_RESULT_LENGTH] + "\n... (truncated)"
                        )
                except Exception as exc:
                    self._logger.error("Tool execution failed: %s", exc)
                    result_text = f"Error: Tool execution failed: {str(exc)[:300]}"

                self._add_history("user", result_text)
                messages.append({"role": "user", "content": result_text})
                await self._send_callback_message(
                    callbacks, iter_num,
                    thought=parsed["thought"], action=action, result=result_text,
                )

            else:
                err = (
                    f"Unknown tool: {tool_name}. "
                    f"Only 'route' and 'execute-tool' are supported."
                )
                self._logger.warning(err)
                self._add_history("user", f"Error: {err}")
                messages.append({"role": "user", "content": f"Error: {err}"})

        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response="I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations.",
            trace_id=tracer.trace_id,
        )

    # ------------------------------------------------------------------
    # Logging / callbacks – mirrors ReAct._send_callback_message
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_callback_message(
        callbacks,
        iter_num: int,
        thought: str = "",
        action=None,
        result: str = "",
        answer: str = "",
    ) -> None:
        logs = []
        if thought:
            logs.append(("thought", thought))
        if action:
            action_str = (
                json.dumps(action, indent=2) if isinstance(action, dict) else str(action)
            )
            logs.append(("action", action_str))
        if result:
            logs.append(("result", result))
        if answer:
            logs.append(("answer", answer))

        data = OrderedDict({"Iteration": iter_num + 1})
        for tag, value in logs:
            data[tag] = value
        send_message(callbacks, message=CallbackMessage(
            source=__file__, type=MessageType.LOG, data=data,
        ))

        plain_text_lines = [
            f"{'=' * 66}\n",
            f"Iteration: {iter_num + 1}\n",
            f"{'-' * 66}\n",
        ]
        for tag, value in logs:
            plain_text_lines.append(f"\033[32m{tag.capitalize()}: {value}\n\n\033[0m")
        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=__file__,
                type=MessageType.LOG,
                metadata={"event": "plain_text", "data": "".join(plain_text_lines)},
            ),
        )
