from __future__ import annotations

import os
import queue
import threading
import time

from google import genai
from google.genai import types as genai_types

from ..swarm.models import ModelCaller, ModelResult


class GeminiCaller:
    """ModelCaller implementation for Google Gemini models."""

    def __init__(self, model: str = "gemini-3.1-flash-lite",
                 api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", "180000"))
        self.timeout_s = max(1.0, timeout_ms / 1000)
        self.timeout_attempts = max(1, int(os.environ.get("GEMINI_TIMEOUT_ATTEMPTS", "1")))
        http_options = genai_types.HttpOptions(timeout=timeout_ms)
        self.client = (
            genai.Client(api_key=key, http_options=http_options)
            if key else genai.Client(http_options=http_options)
        )
        self.model = model

    def _generate_content_with_timeout(
        self,
        prompt: str,
        config: genai_types.GenerateContentConfig,
    ):
        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                result_queue.put(("error", exc))
                return
            result_queue.put(("ok", response))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            kind, value = result_queue.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Gemini call exceeded {int(self.timeout_s)}s watchdog timeout"
            ) from exc
        if kind == "error":
            raise value
        return value

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = genai_types.GenerateContentConfig(**config_kwargs)

        last_err: Exception | None = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                response = self._generate_content_with_timeout(prompt, config)
            except TimeoutError as e:
                last_err = e
                if attempt + 1 < self.timeout_attempts:
                    continue
                raise RuntimeError(f"Gemini call timed out after {attempt + 1} attempts: {e}") from e
            except Exception as e:
                msg = str(e).lower()
                if any(
                    k in msg
                    for k in (
                        "rate",
                        "429",
                        "504",
                        "503",
                        "502",
                        "500",
                        # intermittent provider-side 403s on new models are
                        # transient capacity denials, not real auth failures
                        "403",
                        "permission_denied",
                        "quota",
                        "resource",
                        "deadline",
                        "timeout",
                    )
                ):
                    wait = min(60, 5 * (2 ** attempt))
                    time.sleep(wait)
                    last_err = e
                    continue
                raise
            latency_ms = int((time.perf_counter() - t0) * 1000)

            try:
                text = (response.text or "").strip()
            except ValueError:
                text = ""
            if not text and attempt < 4:
                last_err = ValueError("empty or blocked response")
                time.sleep(2)
                continue

            usage = getattr(response, "usage_metadata", None)
            tokens_in = getattr(usage, "prompt_token_count", 0) or 0
            tokens_out = getattr(usage, "candidates_token_count", 0) or 0

            return ModelResult(
                text=text or "",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_total=tokens_in + tokens_out,
                model=self.model,
                latency_ms=latency_ms,
            )

        raise RuntimeError(f"Gemini call failed after 5 retries: {last_err}")
