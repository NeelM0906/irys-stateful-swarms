from __future__ import annotations

import os
import time

import openai

from ..swarm.models import ModelCaller, ModelResult


class OpenAICaller:
    """ModelCaller implementation for OpenAI models."""

    def __init__(self, model: str = "gpt-5.5",
                 api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=key)
        self.model = model

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            messages.insert(0, {
                "role": "system",
                "content": "Respond ONLY with valid JSON. No markdown fences, "
                           "no explanation, no text outside the JSON object.",
            })

        last_err: Exception | None = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
                last_err = e
                continue
            except openai.APIStatusError as e:
                if e.status_code in (500, 502, 503, 504, 529):
                    wait = min(60, 5 * (2 ** attempt))
                    time.sleep(wait)
                    last_err = e
                    continue
                raise
            except openai.APIConnectionError as e:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
                last_err = e
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)

            choice = response.choices[0] if response.choices else None
            text = (choice.message.content or "").strip() if choice else ""

            if not text and attempt < 4:
                last_err = ValueError("empty response")
                time.sleep(2)
                continue

            if choice and choice.finish_reason == "content_filter":
                last_err = RuntimeError(
                    f"OpenAI content filter on {self.model}: {text[:200]}"
                )
                if attempt < 4:
                    time.sleep(2)
                    continue
                raise last_err

            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0

            return ModelResult(
                text=text or "",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_total=tokens_in + tokens_out,
                model=self.model,
                latency_ms=latency_ms,
                finish_reason=choice.finish_reason if choice else "stop",
                provider_attempts=attempt + 1,
            )

        raise RuntimeError(f"OpenAI call failed after 5 retries: {last_err}")
