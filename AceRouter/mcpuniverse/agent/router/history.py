"""
Format an agent's conversation history into a compact string for the router.

Two versions are supported (see :class:`RouterConfig.history_version`):
    * ``v1`` – only includes ``execute-tool`` actions and their results;
      ``route`` actions are compressed (their ``query`` becomes the user query
      of that round; the ``route`` call itself and its result are dropped).
      Matches the training data format in ``first_10_entries.json``.
    * ``v2`` – keeps both ``route`` and ``execute-tool`` calls.

Output format::

    User: <query>
    Assistant: <think>...</think>
    <tool_call>tool_name{"param": "value"}</tool_call>
    Tool results: [{"name": "tool_name", "results": {...}}]
"""
import json
from typing import Dict, List


# Truncate tool results in history to keep router input size bounded.
MAX_TOOL_RESULT_LENGTH: int = 400

# Keep only the last N rounds in the formatted history.
MAX_ROUNDS_KEPT: int = 1

# Within the retained round, keep only the last N tool call / result pairs.
MAX_ITERATIONS_PER_ROUND: int = 2


def _new_round() -> Dict:
    return {
        "user_query": "",
        "thought": "",
        "tool_calls": [],
        "tool_results": [],
        "final_response": "",
    }


def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_RESULT_LENGTH:
        return text[:MAX_TOOL_RESULT_LENGTH] + "... (truncated)"
    return text


def _render_rounds(rounds: List[Dict], original_query: str) -> str:
    """Render the list of round dicts into the wire format."""
    if rounds and original_query and not rounds[0]["user_query"]:
        rounds[0]["user_query"] = original_query

    lines: List[str] = []
    for rd in rounds:
        if rd["user_query"]:
            lines.append(f"User: {rd['user_query']}")

        if rd["tool_calls"]:
            if rd["thought"]:
                lines.append(f"Assistant: <think>{rd['thought']}</think>")
            else:
                lines.append("Assistant:")
            for tc in rd["tool_calls"]:
                lines.append(
                    f"<tool_call>{tc['tool']}{json.dumps(tc['params'])}</tool_call>"
                )
            if rd["tool_results"]:
                lines.append(
                    f"Tool results: {json.dumps(rd['tool_results'], indent=2)}"
                )

    return "\n".join(lines)


def _trim_rounds(rounds: List[Dict]) -> List[Dict]:
    """Keep only the most recent rounds / iterations."""
    if not rounds:
        return rounds
    latest = rounds[-1]
    if len(latest["tool_calls"]) > MAX_ITERATIONS_PER_ROUND:
        latest["tool_calls"] = latest["tool_calls"][-MAX_ITERATIONS_PER_ROUND:]
    if len(latest["tool_results"]) > MAX_ITERATIONS_PER_ROUND:
        latest["tool_results"] = latest["tool_results"][-MAX_ITERATIONS_PER_ROUND:]
    return rounds[-MAX_ROUNDS_KEPT:]


class HistoryFormatter:
    """Format agent history into a router-friendly string."""

    @staticmethod
    def format(
        history: List[Dict[str, str]],
        original_query: str = "",
        version: str = "v1",
    ) -> str:
        """
        Format ``history`` into a compact string for the router.

        Args:
            history: The agent's conversation history, a list of
                ``{"role": "assistant"|"user", "content": str}`` entries.
            original_query: The user's original question, used when the
                first round does not have an explicit ``route`` query.
            version: ``"v1"`` (default) or ``"v2"``.

        Returns:
            Formatted history string. Empty string if ``history`` is empty.
        """
        if not history:
            return ""
        if version == "v2":
            return HistoryFormatter._format_v2(history, original_query)
        return HistoryFormatter._format_v1(history, original_query)

    # ------------------------------------------------------------------
    # v1 – only execute-tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def _format_v1(history: List[Dict], original_query: str) -> str:
        rounds: List[Dict] = []
        current = _new_round()

        i = 0
        while i < len(history):
            item = history[i]
            role = item.get("role", "")
            content = (item.get("content") or "").strip()

            if role == "assistant":
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # Not a structured assistant response – treat as free-text answer.
                    current["final_response"] = content
                    i += 1
                    continue

                thought = parsed.get("thought", "")
                action = parsed.get("action") or {}
                answer = parsed.get("answer")

                if action:
                    tool_name = action.get("tool", "")
                    arguments = action.get("arguments", {}) or {}

                    if tool_name == "execute-tool":
                        target = arguments.get("tool_name", "")
                        params = arguments.get("params", {}) or {}
                        if thought and not current["thought"]:
                            current["thought"] = thought
                        current["tool_calls"].append({"tool": target, "params": params})

                        if i + 1 < len(history) and history[i + 1].get("role") == "user":
                            result_text = (history[i + 1].get("content") or "").strip()
                            current["tool_results"].append({
                                "name": target,
                                "results": {"result": _truncate(result_text)},
                            })
                            i += 2
                            continue

                    elif tool_name == "route":
                        route_query = arguments.get("query", "")
                        if route_query and not current["user_query"]:
                            current["user_query"] = route_query
                        # v1: drop the route result if present.
                        if i + 1 < len(history) and history[i + 1].get("role") == "user":
                            nxt = history[i + 1].get("content") or ""
                            if "Found" in nxt and "relevant tools" in nxt:
                                i += 2
                                continue
                        i += 1
                        continue

                if answer is not None:
                    current["final_response"] = answer if isinstance(answer, str) else json.dumps(answer)
                    rounds.append(current)
                    current = _new_round()

            i += 1

        if current["tool_calls"] or current["final_response"]:
            rounds.append(current)

        return _render_rounds(_trim_rounds(rounds), original_query)

    # ------------------------------------------------------------------
    # v2 – route + execute-tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def _format_v2(history: List[Dict], original_query: str) -> str:
        rounds: List[Dict] = []
        current = _new_round()

        i = 0
        while i < len(history):
            item = history[i]
            role = item.get("role", "")
            content = (item.get("content") or "").strip()

            if role == "assistant":
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    current["final_response"] = content
                    i += 1
                    continue

                thought = parsed.get("thought", "")
                action = parsed.get("action") or {}
                answer = parsed.get("answer")

                if action:
                    tool_name = action.get("tool", "")
                    arguments = action.get("arguments", {}) or {}

                    if tool_name == "execute-tool":
                        target = arguments.get("tool_name", "")
                        params = arguments.get("params", {}) or {}
                        if thought and not current["thought"]:
                            current["thought"] = thought
                        current["tool_calls"].append({"tool": target, "params": params})
                        if i + 1 < len(history) and history[i + 1].get("role") == "user":
                            result_text = (history[i + 1].get("content") or "").strip()
                            current["tool_results"].append({
                                "name": target,
                                "results": {"result": _truncate(result_text)},
                            })
                            i += 2
                            continue

                    elif tool_name == "route":
                        route_query = arguments.get("query", "")
                        if route_query and not current["user_query"]:
                            current["user_query"] = route_query
                        if thought and not current["thought"]:
                            current["thought"] = thought
                        current["tool_calls"].append({
                            "tool": "route",
                            "params": {"query": route_query},
                        })
                        if i + 1 < len(history) and history[i + 1].get("role") == "user":
                            result_text = (history[i + 1].get("content") or "").strip()
                            current["tool_results"].append({
                                "name": "route",
                                "results": {"result": _truncate(result_text)},
                            })
                            i += 2
                            continue

                if answer is not None:
                    current["final_response"] = answer if isinstance(answer, str) else json.dumps(answer)
                    rounds.append(current)
                    current = _new_round()

            i += 1

        if current["tool_calls"] or current["final_response"]:
            rounds.append(current)

        return _render_rounds(_trim_rounds(rounds), original_query)
