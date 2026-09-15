"""Optional OpenRouter fallback, restricted to free inference."""

from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)


def openrouter_is_configured():
    return bool(
        OPENROUTER_API_KEY
    ) and (
        OPENROUTER_MODEL == "openrouter/free"
        or OPENROUTER_MODEL.endswith(":free")
    )


def build_openrouter_client():
    if not openrouter_is_configured():
        return None

    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=30.0,
        max_retries=0,
    )


openrouter_client = (
    build_openrouter_client()
)


async def request_openrouter_completion(
    *,
    messages,
    tools=None,
    tool_choice="auto",
    max_completion_tokens=1500,
    temperature=None,
    reasoning_format=None,
    reasoning_effort=None,
):
    if openrouter_client is None or not openrouter_is_configured():
        return None

    kwargs = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": (
            max_completion_tokens
        ),
        "extra_body": {
            "provider": {
                "require_parameters": True,
                "max_price": {"prompt": 0, "completion": 0, "request": 0, "image": 0},
            },
            # Groq's reasoning_format/effort options are not portable. Exclude
            # reasoning through OpenRouter without requiring models to disable it.
            "reasoning": {"exclude": True},
        },
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    response = await (
        openrouter_client
        .chat
        .completions
        .create(
            **kwargs
        )
    )
    if not getattr(response, "choices", None):
        raise ValueError("Secondary AI returned no choices.")
    message = response.choices[0].message
    if not (getattr(message, "content", None) or getattr(message, "tool_calls", None)):
        raise ValueError("Secondary AI returned an empty response.")
    return response
