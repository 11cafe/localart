from typing import Annotated, Any, Dict

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool  # type: ignore
from pydantic import BaseModel, Field
from tools.music_providers.minimax_provider import MiniMaxMusicProvider


class GenerateMusicByMiniMaxInputSchema(BaseModel):
    model: str = Field(
        default="music-3.0",
        description=(
            "Optional. The music model to use. Generation models: music-3.0, "
            "music-2.6, music-3.0-free, music-2.6-free. Cover models: "
            "music-cover, music-cover-free. Defaults to music-3.0."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "Optional. A description of the music, specifying style, mood, and "
            "scenario. Required for cover models and for instrumental generation."
        ),
    )
    lyrics: str | None = Field(
        default=None,
        description=(
            "Optional. Song lyrics, using newlines to separate lines. Supports "
            "structure tags such as [Verse], [Chorus], [Bridge], [Outro]. "
            "Required for non-instrumental generation unless lyrics_optimizer is "
            "enabled."
        ),
    )
    output_format: str = Field(
        default="url",
        description=(
            "Optional. The output format of the audio. Allowed values: url, hex. "
            "url links expire after 24 hours."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Optional. Whether to use streaming output. Streaming only supports "
            "the hex output format."
        ),
    )
    is_instrumental: bool = Field(
        default=False,
        description="Optional. Whether to generate instrumental music without vocals.",
    )
    lyrics_optimizer: bool = Field(
        default=False,
        description=(
            "Optional. Whether to automatically generate lyrics from the prompt "
            "when lyrics are not provided."
        ),
    )
    audio_setting: Dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional. Audio output settings, e.g. "
            "{'format': 'mp3', 'sample_rate': 44100, 'bitrate': 128000}. "
            "Allowed formats: mp3, wav, pcm."
        ),
    )
    audio_url: str | None = Field(
        default=None,
        description=(
            "Optional. URL of the reference audio for music-cover generation. "
            "Exactly one reference source is required for cover models."
        ),
    )
    audio_base64: str | None = Field(
        default=None,
        description=(
            "Optional. Base64-encoded reference audio for music-cover generation. "
            "Exactly one reference source is required for cover models."
        ),
    )
    cover_feature_id: str | None = Field(
        default=None,
        description=(
            "Optional. Feature id returned by the music cover preprocess API for "
            "two-step cover generation. Requires lyrics when used."
        ),
    )
    aigc_watermark: bool | None = Field(
        default=None,
        description="Optional. Whether to add an AIGC watermark. Only supported on the CN endpoint.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "generate_music_by_minimax_jaaz",
    description=(
        "Generate music using MiniMax models. Creates original tracks from a "
        "prompt and lyrics with generation models (music-3.0, music-2.6 and free "
        "variants), or creates covers of a reference audio with cover models "
        "(music-cover, music-cover-free). Returns the audio as a url or as "
        "hex-encoded audio."
    ),
    args_schema=GenerateMusicByMiniMaxInputSchema,
)
async def generate_music_by_minimax_jaaz(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    model: str = "music-3.0",
    prompt: str | None = None,
    lyrics: str | None = None,
    output_format: str = "url",
    stream: bool = False,
    is_instrumental: bool = False,
    lyrics_optimizer: bool = False,
    audio_setting: Dict[str, Any] | None = None,
    audio_url: str | None = None,
    audio_base64: str | None = None,
    cover_feature_id: str | None = None,
    aigc_watermark: bool | None = None,
) -> str:
    """
    Generate music using MiniMax models.
    """
    print(f"🎵 MiniMax Music Generation tool_call_id: {tool_call_id}")

    provider = MiniMaxMusicProvider()
    audio = await provider.generate(
        model=model,
        prompt=prompt,
        lyrics=lyrics,
        stream=stream,
        output_format=output_format,
        audio_setting=audio_setting,
        lyrics_optimizer=lyrics_optimizer,
        is_instrumental=is_instrumental,
        audio_url=audio_url,
        audio_base64=audio_base64,
        cover_feature_id=cover_feature_id,
        aigc_watermark=aigc_watermark,
    )

    if output_format == "hex":
        return (
            f"Music generated successfully with {model} as hex-encoded audio "
            f"({len(audio)} hex characters).\n{audio}"
        )
    return f"Music generated successfully with {model}. Audio url (expires in 24 hours):\n{audio}"


__all__ = ["generate_music_by_minimax_jaaz"]
