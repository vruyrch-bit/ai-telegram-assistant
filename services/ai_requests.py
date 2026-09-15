"""Bound completion retries with optional cloud fallback."""

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import math

import groq

from services.cloud_fallback import (
    openrouter_is_configured,
    request_openrouter_completion,
)


logger = logging.getLogger(__name__)


class AIRequestError(Exception):
    """Safe user-facing AI error."""


def _retry_delay(error):
    response = getattr(
        error,
        "response",
        None,
    )

    headers = (
        response.headers
        if response is not None
        else {}
    )

    try:
        if "retry-after-ms" in headers:
            delay = (
                float(
                    headers[
                        "retry-after-ms"
                    ]
                )
                / 1000
            )

        elif "retry-after" in headers:
            value = headers[
                "retry-after"
            ]

            try:
                delay = float(value)

            except ValueError:
                when = parsedate_to_datetime(
                    value
                )

                if when.tzinfo is None:
                    when = when.replace(
                        tzinfo=timezone.utc
                    )

                delay = (
                    when
                    - datetime.now(
                        timezone.utc
                    )
                ).total_seconds()

        else:
            delay = 1.0

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 1.0

    if not math.isfinite(delay):
        return None

    return max(
        0.0,
        delay,
    )


async def _try_openrouter(
    kwargs,
):
    if not openrouter_is_configured():
        return None

    logger.warning(
        "Trying secondary cloud AI provider"
    )

    fallback_kwargs = dict(
        kwargs
    )

    fallback_kwargs.pop(
        "model",
        None,
    )

    try:
        response = (
            await request_openrouter_completion(
                **fallback_kwargs
            )
        )

        if response is not None:
            logger.info(
                "Secondary cloud AI provider "
                "completed request"
            )

        return response

    except Exception as error:
        logger.warning(
            "Secondary cloud AI provider "
            "failed error=%s",
            type(error).__name__,
        )

        return None


async def request_completion(
    client,
    provider_state=None,
    **kwargs,
):
    if provider_state is None:
        provider_state = {}

    # Once one conversation falls back to
    # OpenRouter, keep using OpenRouter for
    # every remaining tool round.
    if (
        provider_state.get("provider")
        == "openrouter"
    ):
        fallback_response = (
            await _try_openrouter(
                kwargs
            )
        )

        if fallback_response is not None:
            return fallback_response

        raise AIRequestError(
            "The secondary AI service is "
            "temporarily unavailable. "
            "Please try again shortly."
        )

    last_status = None

    for attempt in range(2):
        try:
            return await (
                client
                .chat
                .completions
                .create(
                    **kwargs
                )
            )

        except groq.APIError as error:
            status = getattr(
                error,
                "status_code",
                None,
            )

            last_status = status

            logger.warning(
                "AI request failed "
                "error=%s status=%s",
                type(error).__name__,
                status,
            )

            temporary = (
                isinstance(
                    error,
                    groq.APIConnectionError,
                )
                or status in {
                    408,
                    429,
                }
                or (
                    status is not None
                    and status >= 500
                )
            )

            if not temporary:
                body = (
                    error.body
                    if isinstance(
                        error.body,
                        dict,
                    )
                    else {}
                )

                detail = body.get(
                    "error",
                    body,
                )

                code = (
                    detail.get("code")
                    if isinstance(
                        detail,
                        dict,
                    )
                    else None
                )

                if (
                    status == 400
                    and code
                    == "tool_use_failed"
                ):
                    raise AIRequestError(
                        "The AI couldn't form "
                        "a valid action. Please "
                        "rephrase your request."
                    ) from None

                raise

            delay = _retry_delay(
                error
            )

            if (
                attempt == 0
                and delay is not None
                and delay <= 5
            ):
                await asyncio.sleep(
                    delay
                )

                continue

            break

    fallback_response = (
        await _try_openrouter(
            kwargs
        )
    )

    if fallback_response is not None:
        provider_state[
            "provider"
        ] = "openrouter"

        logger.info(
            "AI conversation switched to "
            "secondary cloud provider"
        )

        return fallback_response

    if last_status == 429:
        message = (
            "The AI rate limit has been "
            "reached. Please try again "
            "shortly."
        )

    else:
        message = (
            "The AI service is temporarily "
            "unavailable. Please try again "
            "shortly."
        )

    raise AIRequestError(
        message
    )
