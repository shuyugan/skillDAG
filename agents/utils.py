"""Shared utilities for agents — LLM calling, JSON extraction, etc."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Wrapper around an LLM completion result."""
    content: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}


def call_llm(
    prompt: str,
    *,
    system: str = "",
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Call an LLM with a simple text prompt. Supports OpenAI and Anthropic.

    Provider is auto-detected from the model name:
      - starts with "claude"  → Anthropic (needs ANTHROPIC_API_KEY)
      - starts with "gpt" / "o1" / "o3" / "o4" → OpenAI (needs OPENAI_API_KEY)

    Note: gpt-5 / o1 / o3 / o4-mini models do NOT support temperature / top_p;
    these parameters are automatically omitted for those models.
    """
    if model.startswith("claude"):
        return _call_anthropic(prompt, system=system, model=model,
                               temperature=temperature, max_tokens=max_tokens)
    elif model.startswith(("gpt", "o1", "o3", "o4")):
        return _call_openai(prompt, system=system, model=model,
                            temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(
            f"Unknown model prefix: {model!r}. "
            "Expected model name starting with 'claude', 'gpt', 'o1', 'o3', or 'o4'."
        )


# Models that do NOT support temperature / top_p
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_anthropic(
    prompt: str, *, system: str, model: str,
    temperature: float, max_tokens: int,
) -> LLMResponse:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system if system else anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    return LLMResponse(
        content=msg.content[0].text,
        model=msg.model,
        usage={
            "prompt_tokens": msg.usage.input_tokens,
            "completion_tokens": msg.usage.output_tokens,
            "total_tokens": msg.usage.input_tokens + msg.usage.output_tokens,
        },
    )


def _call_openai(
    prompt: str, *, system: str, model: str,
    temperature: float, max_tokens: int,
) -> LLMResponse:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"model": model, "messages": messages}

    # o1/o3/o4/gpt-5 do not support temperature, top_p, or max_tokens;
    # they use max_completion_tokens instead.
    if model.startswith(_NO_TEMPERATURE_PREFIXES):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    usage = resp.usage
    return LLMResponse(
        content=choice.message.content,
        model=resp.model,
        usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    )


# ---------------------------------------------------------------------------
# JSON extraction from LLM responses
# ---------------------------------------------------------------------------

def extract_json(raw_text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles responses wrapped in ```json ... ``` fences or bare JSON.
    Raises ValueError if no JSON object can be found or parsed.
    """
    # Try ```json ... ``` fenced block first
    json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try bare JSON object
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"Could not extract JSON from LLM response:\n{raw_text[:500]}")
