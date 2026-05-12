"""End-to-end tests for DynamicReAct with mocked LLM and MCP tools."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcp.types import CallToolResult, TextContent

from mcpuniverse.agent.dynamic_react import DynamicReAct, DynamicReActConfig
from mcpuniverse.agent.router import RouterConfig


def _tool(name: str, description: str, schema=None):
    return SimpleNamespace(
        name=name, description=description, inputSchema=schema or {"type": "object"},
    )


def _ok_tool_result(text: str):
    return CallToolResult(content=[TextContent(type="text", text=text)])


class TestDynamicReActExecuteFlow(unittest.IsolatedAsyncioTestCase):
    """Exercise the full reasoning loop (route + execute-tool + answer) with mocks."""

    def _make_agent(self, responses, tool_result_text='{"ok": true}'):
        cfg = DynamicReActConfig(
            max_iterations=10,
            prompt_version="v1",
            router=RouterConfig(mode="embedding", embedding_model="bm25", max_tools=2),
        )
        agent = DynamicReAct(
            mcp_manager=MagicMock(),
            llm=MagicMock(),
            config=cfg.to_dict(),
        )
        # Fake self._tools normally set by BaseAgent.initialize.
        agent._tools = {
            "math": [
                _tool("add", "add two numbers",
                      {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}),
                _tool("mul", "multiply two numbers", {"type": "object"}),
            ],
        }
        # Patch LLM to replay scripted responses.
        idx = [0]

        async def gen(messages, tracer=None, callbacks=None, **kw):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        agent._llm.generate_async = gen

        # Patch tool execution.
        async def call_tool(action, tracer=None, callbacks=None):
            return _ok_tool_result(tool_result_text)

        agent.call_tool = call_tool
        return agent, idx

    async def test_route_then_execute_then_answer(self):
        responses = [
            json.dumps({"thought": "need tool",
                         "action": {"tool": "route", "arguments": {"query": "add numbers"}}}),
            json.dumps({"thought": "call add",
                         "action": {"tool": "execute-tool",
                                    "arguments": {"server_name": "math",
                                                  "tool_name": "add",
                                                  "params": {"a": 3, "b": 5}}}}),
            json.dumps({"thought": "done", "answer": "result is 8"}),
        ]
        agent, idx = self._make_agent(responses, tool_result_text='{"result": 8}')
        resp = await agent._execute("3+5?")
        self.assertEqual(resp.response, "result is 8")
        self.assertEqual(idx[0], 3)

    async def test_unknown_tool_name_is_error_branch(self):
        responses = [
            json.dumps({"thought": "go", "action": {"tool": "unknown_thing", "arguments": {}}}),
            json.dumps({"thought": "give up", "answer": "ok"}),
        ]
        agent, _ = self._make_agent(responses)
        resp = await agent._execute("x")
        self.assertEqual(resp.response, "ok")
        # Ensure the error was recorded in history
        joined = " ".join(h["content"] for h in agent.get_history())
        self.assertIn("Unknown tool", joined)

    async def test_max_iterations_exhausted(self):
        # Always emit an action that can't produce a final answer
        bad = json.dumps({"thought": "t", "action": {"tool": "nonexistent", "arguments": {}}})
        cfg = DynamicReActConfig(
            max_iterations=3,
            router=RouterConfig(mode="embedding", embedding_model="bm25"),
        )
        agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
        agent._tools = {}

        async def gen(messages, tracer=None, callbacks=None, **kw):
            return bad

        agent._llm.generate_async = gen

        resp = await agent._execute("hi")
        self.assertIn("couldn't find", resp.response.lower())

    async def test_router_hallucinated_tool_is_silently_dropped(self):
        # Router returns a tool name that doesn't exist in self._tools.
        # The formatter should produce "Found 0 relevant tools:" and the LLM should pick another path.
        responses = [
            json.dumps({"thought": "route",
                         "action": {"tool": "route", "arguments": {"query": "something"}}}),
            json.dumps({"thought": "no usable tool, just answer",
                         "answer": "no tools available"}),
        ]
        cfg = DynamicReActConfig(
            max_iterations=5,
            router=RouterConfig(mode="embedding", embedding_model="bm25", max_tools=1),
        )
        agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
        agent._tools = {"s": [_tool("real_tool", "a real tool")]}
        idx = [0]

        async def gen(messages, tracer=None, callbacks=None, **kw):
            r = responses[idx[0]]; idx[0] += 1
            return r
        agent._llm.generate_async = gen

        # Force router to return a hallucinated name by mocking the backend.
        async def bad_route(**kwargs):
            return ["never_existed_tool"]
        agent._router._backend.route = bad_route

        resp = await agent._execute("x")
        # The route result should show 0 tools (hallucination dropped)
        joined = " ".join(h["content"] for h in agent.get_history())
        self.assertIn("Found 0 relevant tools", joined)
        self.assertEqual(resp.response, "no tools available")

    async def test_empty_tools_does_not_crash_route(self):
        responses = [
            json.dumps({"thought": "route",
                         "action": {"tool": "route", "arguments": {"query": "q"}}}),
            json.dumps({"thought": "give up", "answer": "no tools"}),
        ]
        cfg = DynamicReActConfig(
            max_iterations=5,
            router=RouterConfig(mode="embedding", embedding_model="bm25"),
        )
        agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
        agent._tools = {}  # no servers
        idx = [0]

        async def gen(messages, tracer=None, callbacks=None, **kw):
            r = responses[idx[0]]; idx[0] += 1
            return r
        agent._llm.generate_async = gen

        resp = await agent._execute("x")
        self.assertEqual(resp.response, "no tools")

    async def test_malformed_json_from_llm_does_not_crash(self):
        responses = [
            "not a json at all",
            json.dumps({"thought": "retry with good json", "answer": "recovered"}),
        ]
        agent, _ = self._make_agent(responses)
        resp = await agent._execute("x")
        self.assertEqual(resp.response, "recovered")

    async def test_prompt_version_v1_and_fallback(self):
        """v1 loads directly; unknown versions (like v2/v3) fall back to v1 with a warning."""
        for ver in ("v1", "v2", "v3", "unknown"):
            cfg = DynamicReActConfig(
                prompt_version=ver,
                max_iterations=1,
                router=RouterConfig(mode="embedding", embedding_model="bm25"),
            )
            agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
            agent._tools = {}
            prompt = agent._build_prompt("the question")
            self.assertIn("the question", prompt)

    async def test_prompt_version_react_template_works(self):
        """The 'react' prompt version should work too (different template)."""
        cfg = DynamicReActConfig(
            prompt_version="react",
            max_iterations=1,
            router=RouterConfig(mode="embedding", embedding_model="bm25"),
        )
        agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
        agent._tools = {}
        prompt = agent._build_prompt("the question")
        self.assertIn("the question", prompt)

    async def test_custom_prompt_template_override(self):
        """system_prompt path overrides prompt_version."""
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".j2", delete=False) as fh:
            fh.write("CUSTOM: {{QUESTION}}")
            path = fh.name
        try:
            cfg = DynamicReActConfig(
                system_prompt=path,
                max_iterations=1,
                router=RouterConfig(mode="embedding", embedding_model="bm25"),
            )
            agent = DynamicReAct(mcp_manager=MagicMock(), llm=MagicMock(), config=cfg.to_dict())
            agent._tools = {}
            self.assertEqual(agent._build_prompt("hello"), "CUSTOM: hello")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
