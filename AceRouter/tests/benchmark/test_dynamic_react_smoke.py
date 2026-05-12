"""
Smoke tests for :class:`DynamicReAct` integration with :class:`BenchmarkRunner`.

These tests verify that the benchmark YAML → WorkflowBuilder → agent
instantiation path works end-to-end for both the MCP-Universe and MCPMark
benchmark configs. They do **not** make real LLM calls; they stop just before
``agent.initialize()`` (which would start MCP subprocesses) and before
``agent.execute()`` (which would hit the reasoning LLM).

To run a real benchmark end-to-end, set ``OPENAI_API_KEY`` and call
``BenchmarkRunner(...).run(...)``.
"""
import os
import unittest

import pytest

from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.agent.dynamic_react import DynamicReAct
from mcpuniverse.agent.router import ToolRouter
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.workflows.builder import WorkflowBuilder


class _BuildOnlyMixin:
    """Common helpers for benchmark-config sanity tests."""

    config_path: str = ""  # override

    def _load_and_build(self):
        runner = BenchmarkRunner(self.config_path)
        mcp_manager = MCPManager()
        workflow = WorkflowBuilder(
            mcp_manager=mcp_manager, config=runner._agent_configs  # pylint: disable=protected-access
        )
        workflow.build()
        return runner, workflow

    def _assert_agent(self, workflow, agent_name: str):
        agent = workflow.get_component(agent_name)
        assert isinstance(agent, DynamicReAct), (
            f"Expected DynamicReAct, got {type(agent).__name__}"
        )
        assert isinstance(agent._router, ToolRouter), "Router not attached"  # pylint: disable=protected-access
        # Router config nested correctly
        assert agent._config.router.mode == "llm"  # pylint: disable=protected-access
        # Router LLM built via ModelManager
        llm_backend = agent._router._backend  # pylint: disable=protected-access
        assert hasattr(llm_backend, "_llm"), "LLM router backend missing ._llm"
        return agent


class TestDynamicReActMCPUniverseConfig(unittest.TestCase, _BuildOnlyMixin):
    """DynamicReAct on the MCP-Universe side (weather smoke test)."""

    config_path = "mcpuniverse/configs/smoke_dynamic_react.yaml"

    def test_config_loads_and_builds(self):
        runner, workflow = self._load_and_build()
        # Benchmark metadata
        assert len(runner._benchmark_configs) == 1  # pylint: disable=protected-access
        bm = runner._benchmark_configs[0]  # pylint: disable=protected-access
        assert bm.agent == "dynamic-react-agent"
        assert bm.tasks == ["mcpuniverse/tasks/smoke/weather_1.json"]
        # Agent
        agent = self._assert_agent(workflow, "dynamic-react-agent")
        # servers config
        server_names = [s["name"] for s in agent._config.servers]  # pylint: disable=protected-access
        assert server_names == ["weather"]
        # prompts
        assert agent._config.prompt_version == "v1"  # pylint: disable=protected-access
        assert agent._config.max_iterations == 10  # pylint: disable=protected-access

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY + network",
    )
    def test_full_run_requires_api_key(self):
        """Marker test: only runs when OPENAI_API_KEY is set."""
        import asyncio

        from mcpuniverse.tracer.collectors import MemoryCollector

        config_path = "mcpuniverse/configs/smoke_dynamic_react.yaml"

        async def _run():
            runner = BenchmarkRunner(config_path)
            return await runner.run(trace_collector=MemoryCollector())

        results = asyncio.run(_run())
        assert len(results) == 1


class TestDynamicReActMCPMarkConfig(unittest.TestCase, _BuildOnlyMixin):
    """DynamicReAct on the MCPMark side (filesystem)."""

    config_path = "mcpmark/configs/mcpmark_filesystem_dynamic_react.yaml"

    def test_config_loads_and_builds(self):
        runner, workflow = self._load_and_build()
        bm = runner._benchmark_configs[0]  # pylint: disable=protected-access
        assert bm.agent == "dynamic-react-agent"
        assert len(bm.tasks) == 1
        agent = self._assert_agent(workflow, "dynamic-react-agent")
        # MCPMark tasks typically use use_specified_server: true, so the agent's
        # own `servers` list may be empty – the task will override it at run time.
        assert agent._config.max_iterations == 20  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
