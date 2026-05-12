"""Integration tests for ToolRouter with mocked backends (no real LLM/network)."""
import unittest
from unittest.mock import AsyncMock

from mcpuniverse.agent.router import RouterConfig, ToolRouter


class TestToolRouter(unittest.IsolatedAsyncioTestCase):
    def _build_embedding_router(self, **kwargs):
        cfg = RouterConfig(
            mode="embedding", embedding_model="bm25", max_tools=2, **kwargs,
        )
        return ToolRouter(cfg)

    async def test_embedding_bm25_end_to_end(self):
        router = self._build_embedding_router()
        tools = [
            {"name": "get_weather", "description": "weather forecast for a city"},
            {"name": "send_email", "description": "send an email"},
        ]
        selected = await router.route(
            query="weather in Paris",
            tools_for_routing=tools,
            history=None,
            original_query="weather in Paris",
        )
        self.assertIsInstance(selected, list)
        # BM25 should prefer the weather-related tool (has lexical overlap on "weather")
        self.assertIn("get_weather", selected)

    async def test_shuffle_tools_does_not_mutate_caller_list(self):
        router = self._build_embedding_router(shuffle_tools=True)
        tools = [
            {"name": f"tool_{i}", "description": f"description {i}"}
            for i in range(10)
        ]
        original = list(tools)
        await router.route(query="test", tools_for_routing=tools, history=None)
        self.assertEqual(tools, original, "caller's list was mutated")

    async def test_empty_tools_returns_empty(self):
        router = self._build_embedding_router()
        selected = await router.route(
            query="anything", tools_for_routing=[], history=None,
        )
        self.assertEqual(selected, [])

    async def test_llm_mode_requires_llm_config(self):
        with self.assertRaises(ValueError):
            ToolRouter(RouterConfig(mode="llm"))  # missing 'llm'

    async def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ToolRouter(RouterConfig(mode="banana"))

    async def test_history_forwarded_to_backend(self):
        """With enable_history=True and non-empty history, backend should see a non-empty history string."""
        router = self._build_embedding_router(enable_history=True)
        # Replace backend with an AsyncMock to capture its args
        captured = {}

        async def fake_route(query, tools, history, config, tracer=None, callbacks=None):
            captured["history"] = history
            captured["query"] = query
            return [tools[0]["name"]] if tools else []

        router._backend.route = fake_route  # pylint: disable=protected-access

        hist = [
            {"role": "assistant",
             "content": '{"thought":"t","action":{"tool":"execute-tool","arguments":{"server_name":"x","tool_name":"y","params":{}}}}'},
            {"role": "user", "content": "tool result"},
        ]
        await router.route(
            query="next step",
            tools_for_routing=[{"name": "a", "description": "x"}],
            history=hist,
            original_query="original",
        )
        self.assertNotEqual(captured["history"], "")
        self.assertIn("<tool_call>", captured["history"])

    async def test_disable_history_sends_empty_string(self):
        router = self._build_embedding_router(enable_history=False)
        captured = {}

        async def fake_route(query, tools, history, config, tracer=None, callbacks=None):
            captured["history"] = history
            return []

        router._backend.route = fake_route  # pylint: disable=protected-access

        hist = [{"role": "assistant", "content": "{}"}, {"role": "user", "content": "x"}]
        await router.route(
            query="next", tools_for_routing=[{"name": "a", "description": ""}], history=hist,
        )
        self.assertEqual(captured["history"], "")


if __name__ == "__main__":
    unittest.main()
