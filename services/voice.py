from groq import AsyncGroq

from config import (
    GROQ_API_KEY,
    VOICE_MODEL,
)


voice_client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=30.0,
    max_retries=1,
)


async def transcribe_voice(
    audio_bytes: bytes,
):
    transcription = (
        await voice_client
        .audio
        .transcriptions
        .create(
            file=(
                "voice.ogg",
                audio_bytes,
            ),
            model=VOICE_MODEL,
            response_format="json",
            temperature=0.0,
        )
    )

    return transcription.text
