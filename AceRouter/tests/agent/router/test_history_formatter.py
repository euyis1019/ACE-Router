"""Unit tests for HistoryFormatter (v1 and v2)."""
import json
import unittest

from mcpuniverse.agent.router.history import HistoryFormatter


def _a(**kwargs):
    return {"role": "assistant", "content": json.dumps(kwargs)}


def _u(text):
    return {"role": "user", "content": text}


class TestHistoryFormatter(unittest.TestCase):
    def test_empty_history_returns_empty_string(self):
        self.assertEqual(HistoryFormatter.format([], "any query"), "")

    def test_v1_drops_route_results(self):
        history = [
            _a(thought="need github", action={"tool": "route", "arguments": {"query": "create repo"}}),
            _u("Found 3 relevant tools: ..."),
            _a(thought="call create_repo",
               action={"tool": "execute-tool",
                       "arguments": {"server_name": "github",
                                     "tool_name": "create_repo",
                                     "params": {"name": "test"}}}),
            _u('{"success": true, "url": "https://github.com/x/y"}'),
        ]
        out = HistoryFormatter.format(history, original_query="please create a repo", version="v1")
        self.assertIn("User: create repo", out)
        self.assertIn('<tool_call>create_repo{"name": "test"}</tool_call>', out)
        self.assertIn('Tool results', out)
        self.assertIn('"name": "create_repo"', out)
        # v1 drops the route call:
        self.assertNotIn('<tool_call>route', out)
        self.assertNotIn('"name": "route"', out)

    def test_v2_keeps_route_and_execute(self):
        history = [
            _a(thought="need github", action={"tool": "route", "arguments": {"query": "create repo"}}),
            _u("Found 3 relevant tools: ..."),
            _a(thought="call create_repo",
               action={"tool": "execute-tool",
                       "arguments": {"server_name": "github",
                                     "tool_name": "create_repo",
                                     "params": {"name": "test"}}}),
            _u('{"success": true}'),
        ]
        out = HistoryFormatter.format(history, original_query="...", version="v2")
        self.assertIn('<tool_call>route{"query": "create repo"}</tool_call>', out)
        self.assertIn('<tool_call>create_repo{"name": "test"}</tool_call>', out)
        self.assertIn('"name": "route"', out)
        self.assertIn('"name": "create_repo"', out)

    def test_original_query_used_when_no_route_query(self):
        history = [
            _a(thought="direct execute",
               action={"tool": "execute-tool",
                       "arguments": {"server_name": "x", "tool_name": "y", "params": {}}}),
            _u("result"),
        ]
        out = HistoryFormatter.format(history, original_query="the real query", version="v1")
        self.assertIn("User: the real query", out)

    def test_v1_truncates_long_tool_result(self):
        big = "X" * 2000
        history = [
            _a(thought="t",
               action={"tool": "execute-tool",
                       "arguments": {"server_name": "x", "tool_name": "y", "params": {}}}),
            _u(big),
        ]
        out = HistoryFormatter.format(history, original_query="q", version="v1")
        self.assertIn("... (truncated)", out)
        # Should NOT contain the full 2000-char string
        self.assertNotIn(big, out)

    def test_malformed_assistant_message_does_not_crash(self):
        history = [
            {"role": "assistant", "content": "not valid json"},
            _u("ignore"),
        ]
        # Should not raise
        out = HistoryFormatter.format(history, original_query="q", version="v1")
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
