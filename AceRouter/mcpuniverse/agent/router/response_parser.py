"""
Parse the raw output from a router LLM into a list of selected tool names.

Handles assorted quirks seen in practice:
    * ``<think>...</think>`` reasoning blocks
    * Markdown fences (```` ```json ````, ```` ``` ````)
    * Single-tool strings (``"tool_name"`` instead of an array)
    * Trailing commas
    * Object arrays (``[{"tool": "t1", "reason": "..."}]``)
    * Multiple adjacent arrays (which are merged and deduped)
"""
import json
import re
from typing import Any, List, Optional

from mcpuniverse.common.logger import get_logger

logger = get_logger("ResponseParser")


class ResponseParser:
    """Extract a deduplicated list of tool names from the router's raw response."""

    @staticmethod
    def parse_tool_names(raw_response: str) -> List[str]:
        """
        Parse ``raw_response`` into a list of tool names.

        Returns:
            A list of tool names (order preserved, duplicates removed).
            Returns an empty list if parsing fails.
        """
        if not raw_response:
            return []

        try:
            cleaned = ResponseParser._clean_response(raw_response)
            cleaned = ResponseParser._normalize_single_string(cleaned)
            cleaned = ResponseParser._locate_array_start(cleaned)

            arrays = ResponseParser._extract_arrays(cleaned)
            if not arrays:
                return []

            if len(arrays) == 1:
                routing_results: List[Any] = arrays[0]
            else:
                logger.warning(
                    "Router response contains %d arrays; merging and deduping.",
                    len(arrays),
                )
                merged: List[Any] = []
                for arr in arrays:
                    merged.extend(arr)
                routing_results = list(dict.fromkeys(merged))

            return ResponseParser._normalize_results(routing_results)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to parse router response: %s\nRaw:\n%s", exc, raw_response)
            return []

    # ------------------------------------------------------------------
    # Cleaning / normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response(response: str) -> str:
        text = response.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        # Strip <think>...</think> blocks that some routers (Qwen3, etc.) emit.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    @staticmethod
    def _normalize_single_string(text: str) -> str:
        """Handle ``"tool_name"`` → ``["tool_name"]``."""
        if text.startswith('"') and not text.startswith("["):
            m = re.match(r'^"([^"]+)"', text)
            if m:
                return f'["{m.group(1)}"]'
        return text

    @staticmethod
    def _locate_array_start(text: str) -> str:
        """If no leading ``[`` or ``{``, try to find one or extract a quoted name."""
        if text.startswith("[") or text.startswith("{"):
            return text
        idx = text.find("[")
        if idx != -1:
            return text[idx:]
        # Last resort: find any quoted tool name.
        m = re.search(r'"([^"]+)"', text)
        if m:
            return f'["{m.group(1)}"]'
        raise json.JSONDecodeError("No valid JSON found", text, 0)

    # ------------------------------------------------------------------
    # Array extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_arrays(text: str) -> List[list]:
        """Extract every balanced JSON array in ``text`` (ignore individual failures)."""
        arrays: List[list] = []
        start = 0
        while True:
            array_start = text.find("[", start)
            if array_start == -1:
                break

            depth = 0
            array_end = -1
            for i in range(array_start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        array_end = i
                        break

            if array_end == -1:
                break

            candidate = text[array_start:array_end + 1]
            candidate = re.sub(r",\s*]", "]", candidate)
            try:
                arrays.append(json.loads(candidate))
            except json.JSONDecodeError:
                pass

            start = array_end + 1
        return arrays

    # ------------------------------------------------------------------
    # Normalization to List[str]
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_results(routing_results: List[Any]) -> List[str]:
        """
        Accept either ``["t1", "t2"]`` or ``[{"tool": "t1", "reason": "..."}]``.

        Returns an ordered list of unique tool names.
        """
        names: List[str] = []
        for result in routing_results:
            if isinstance(result, str):
                tool_name: Optional[str] = result
            elif isinstance(result, dict):
                tool_name = result.get("tool") or result.get("name")
            else:
                tool_name = None

            if tool_name:
                names.append(tool_name)

        return list(dict.fromkeys(names))
