"""Unit tests for ResponseParser."""
import unittest

from mcpuniverse.agent.router.response_parser import ResponseParser


class TestResponseParser(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('["t1", "t2"]'),
            ["t1", "t2"],
        )

    def test_quoted_single_string(self):
        self.assertEqual(ResponseParser.parse_tool_names('"t1"'), ["t1"])

    def test_think_tags_stripped(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('<think>reasoning...</think>\n["t1"]'),
            ["t1"],
        )

    def test_json_code_fence(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('```json\n["t1"]\n```'),
            ["t1"],
        )

    def test_plain_code_fence(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('```\n["t1"]\n```'),
            ["t1"],
        )

    def test_object_array(self):
        self.assertEqual(
            ResponseParser.parse_tool_names(
                '[{"tool": "t1", "reason": "x"}, {"tool": "t2", "reason": "y"}]'
            ),
            ["t1", "t2"],
        )

    def test_trailing_comma(self):
        self.assertEqual(ResponseParser.parse_tool_names('["t1",]'), ["t1"])

    def test_duplicates_removed(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('["t1", "t1", "t2"]'),
            ["t1", "t2"],
        )

    def test_multiple_arrays_merged(self):
        self.assertEqual(
            ResponseParser.parse_tool_names('["t1"] ["t2"]'),
            ["t1", "t2"],
        )

    def test_think_plus_code_fence(self):
        raw = """<think>
I need to pick the most relevant tool for weather.
</think>

["get_forecast"]
"""
        self.assertEqual(ResponseParser.parse_tool_names(raw), ["get_forecast"])

    def test_empty_input(self):
        self.assertEqual(ResponseParser.parse_tool_names(""), [])

    def test_garbage_returns_empty(self):
        self.assertEqual(ResponseParser.parse_tool_names("totally invalid"), [])

    def test_prefix_garbage_then_array(self):
        # Some routers emit explanation before the array
        self.assertEqual(
            ResponseParser.parse_tool_names('The best tool is: ["t1"]'),
            ["t1"],
        )


if __name__ == "__main__":
    unittest.main()
