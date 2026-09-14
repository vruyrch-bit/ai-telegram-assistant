"""Bound completion retries without replaying application tools."""
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import logging

import groq

logger = logging.getLogger(__name__)


class AIRequestError(Exception):
    """A safe, user-facing error; never includes provider payloads or secrets."""


def _retry_delay(error):
    response = getattr(error, 'response', None)
    headers = response.headers if response is not None else {}
    try:
        if 'retry-after-ms' in headers:
            delay = float(headers['retry-after-ms']) / 1000
        elif 'retry-after' in headers:
            value = headers['retry-after']
            try:
                delay = float(value)
            except ValueError:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = (when - datetime.now(timezone.utc)).total_seconds()
        else:
            delay = 1.0
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return max(0.0, delay) if math.isfinite(delay) else None


async def request_completion(client, **kwargs):
    for attempt in range(2):
        try:
            return await client.chat.completions.create(**kwargs)
        except groq.APIError as error:
            status = getattr(error, 'status_code', None)
            logger.warning('AI request failed error=%s status=%s', type(error).__name__, status)
            temporary = isinstance(error, groq.APIConnectionError) or status in {408, 429} or (
                status is not None and status >= 500)
            if not temporary:
                body = error.body if isinstance(error.body, dict) else {}
                detail = body.get('error', body)
                code = detail.get('code') if isinstance(detail, dict) else None
                if status == 400 and code == 'tool_use_failed':
                    raise AIRequestError(
                        "The AI couldn't form a valid action. Please rephrase your request."
                    ) from None
                raise
            delay = _retry_delay(error)
            # Respect long server cooldowns by returning promptly, not retrying early.
            if attempt == 0 and delay is not None and delay <= 5:
                await asyncio.sleep(delay)
                continue
            if status == 429:
                message = 'The AI rate limit has been reached. Please try again shortly.'
            else:
                message = 'The AI service is temporarily unavailable. Please try again shortly.'
            raise AIRequestError(message) from None
