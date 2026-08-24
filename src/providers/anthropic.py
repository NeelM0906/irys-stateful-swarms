from __future__ import annotations

import os
import time

import anthropic

from ..swarm.models import ModelCaller, ModelResult


class AnthropicCaller:
    """ModelCaller implementation for Anthropic Claude models."""

    def __init__(self, model: str = "claude-fable-5",
                 api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        system_parts = []
        if json_mode:
            system_parts.append(
                "Respond ONLY with valid JSON. No markdown fences, "
                "no explanation, no text outside the JSON object."
            )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        last_err: Exception | None = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
            except anthropic.RateLimitError as e:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
                last_err = e
                continue
            except anthropic.APIStatusError as e:
                if e.status_code in (500, 502, 503, 504, 529):
                    wait = min(60, 5 * (2 ** attempt))
                    time.sleep(wait)
                    last_err = e
                    continue
                raise
            except anthropic.APIConnectionError as e:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
                last_err = e
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            text = text.strip()

            if not text and attempt < 4:
                last_err = ValueError("empty response")
                time.sleep(2)
                continue

            if getattr(response, "stop_reason", None) == "refusal":
                last_err = RuntimeError(
                    f"Anthropic refusal on {self.model}: {text[:200]}"
                )
                if attempt < 4:
                    time.sleep(2)
                    continue
                raise last_err

            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

            finish = getattr(response, "stop_reason", "end_turn") or "end_turn"

            return ModelResult(
                text=text or "",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_total=tokens_in + tokens_out,
                model=self.model,
                latency_ms=latency_ms,
                finish_reason=finish,
                provider_attempts=attempt + 1,
            )

        raise RuntimeError(f"Anthropic call failed after 5 retries: {last_err}")
