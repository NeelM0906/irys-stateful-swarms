"""LLM call helpers for the loop — one place for call mechanics.

Every call goes through call_json/call_text so usage is always tracked on
the board and every call is visible in the event log.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .state import Board


def call_json(caller, board: Board, prompt: str, *, kind: str,
              max_tokens: int = 8192, temperature: float = 0.1) -> Any:
    """Call the model expecting JSON. Returns parsed object or None."""
    result = caller.complete(
        prompt, max_tokens=max_tokens, temperature=temperature, json_mode=True,
    )
    board.add_tokens(result.tokens_input, result.tokens_output, result.model)
    parsed = parse_json(result.text)
    finish = getattr(result, "finish_reason", "stop") or "stop"
    p_attempts = getattr(result, "provider_attempts", 1) or 1
    board.log(
        kind, f"{kind} call ({result.model}, {result.tokens_total} tok)",
        detail={"parsed": parsed is not None, "finish_reason": finish,
                "provider_attempts": p_attempts},
        model=result.model, tokens=result.tokens_total,
        tokens_in=result.tokens_input, tokens_out=result.tokens_output,
    )
    return parsed


def call_text(caller, board: Board, prompt: str, *, kind: str,
              max_tokens: int = 16384, temperature: float = 0.2) -> str:
    """Call the model expecting prose. Returns text ('' on failure)."""
    result = caller.complete(
        prompt, max_tokens=max_tokens, temperature=temperature, json_mode=False,
    )
    board.add_tokens(result.tokens_input, result.tokens_output, result.model)
    board.log(
        kind, f"{kind} call ({result.model}, {result.tokens_total} tok)",
        model=result.model, tokens=result.tokens_total,
        tokens_in=result.tokens_input, tokens_out=result.tokens_output,
    )
    return result.text


def parse_json(text: str) -> Any:
    """Robustly parse JSON from model output (fences, preambles, trailing junk)."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # First balanced object or array
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None
