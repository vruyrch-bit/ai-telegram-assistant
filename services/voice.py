import asyncio
import io
from functools import lru_cache
from groq import AsyncGroq
from config import AI_PROVIDER

from config import (
    GROQ_API_KEY,
    VOICE_MODEL,
    VOICE_LANGUAGE,
)


voice_client = None if AI_PROVIDER == "local" else AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=30.0,
    max_retries=1,
)


async def transcribe_voice(
    audio_bytes: bytes,
):
    if AI_PROVIDER == "local":
        return await asyncio.to_thread(transcribe_local, audio_bytes)
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
            **({"language": VOICE_LANGUAGE} if VOICE_LANGUAGE else {}),
        )
    )

    return transcription.text


@lru_cache(maxsize=1)
def local_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(VOICE_MODEL, device="cpu", compute_type="int8", cpu_threads=4,
                        download_root=".runtime/whisper")


def transcribe_local(audio_bytes):
    segments, _ = local_whisper().transcribe(io.BytesIO(audio_bytes),
        language=VOICE_LANGUAGE or None, beam_size=3, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments)
