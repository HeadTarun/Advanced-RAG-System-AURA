"""
llm/groq_client.py
------------------
Thin wrapper around the Groq SDK.

Key features:
  - Separate methods for 8B (fast/cheap) and 70B (deep reasoning)
  - JSON mode enforcement via response_format parameter
  - Pydantic schema validation as a second layer
  - Automatic retry with exponential backoff (tenacity)
  - Token usage logging
"""

import json
from typing import Any, Optional, Type

from groq import Groq
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import GROQ_API_KEY, MODEL_8B, MODEL_70B, CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqClient:
    """
    Singleton Groq client.  Use the module-level instance `groq_client`.

    Usage:
        response = groq_client.call_8b(messages, schema=MyPydanticModel)
        response = groq_client.call_70b(messages)  # returns raw str
    """

    def __init__(self) -> None:
        self._client = Groq(api_key=GROQ_API_KEY)

    # ------------------------------------------------------------------
    # Internal low-level call
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        """
        Returns (content_str, usage_dict).
        When json_mode=True, Groq enforces valid JSON in the response.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or ""
        usage = {
            "prompt_tokens":     completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens":      completion.usage.total_tokens,
            "model":             model,
        }
        logger.debug("groq_call", **usage)
        return content, usage

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def call_8b(
        self,
        messages: list[dict],
        json_mode: bool = True,
        schema: Optional[Type[BaseModel]] = None,
        max_tokens: int = 512,
    ) -> Any:
        """
        Call the 8B model.
        If `schema` is provided, parse and validate the JSON response
        into a Pydantic model — raises ValueError on failure.
        """
        content, _ = self._call(
            MODEL_8B, messages, max_tokens, json_mode=json_mode
        )
        if schema:
            return self._parse_schema(content, schema)
        if json_mode:
            return self._safe_json(content)
        return content

    def call_70b(
        self,
        messages: list[dict],
        json_mode: bool = False,
        schema: Optional[Type[BaseModel]] = None,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Call the 70B model.
        Default max_tokens comes from global config if not specified.
        """
        mt = max_tokens or CONFIG.tokens.max_answer_tokens
        content, _ = self._call(
            MODEL_70B, messages, mt, json_mode=json_mode
        )
        if schema:
            return self._parse_schema(content, schema)
        if json_mode:
            return self._safe_json(content)
        return content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(text: str) -> dict:
        """Strip markdown fences then parse JSON."""
        cleaned = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Groq returned invalid JSON: {e}\nContent: {text[:300]}")

    @staticmethod
    def _parse_schema(text: str, schema: Type[BaseModel]) -> BaseModel:
        data = GroqClient._safe_json(text)
        try:
            return schema(**data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")


# Module-level singleton
groq_client = GroqClient()
