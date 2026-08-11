import traceback
from typing import Any, Dict, Optional

from services.config_service import config_service
from utils.http_client import HttpClient

# Generation and cover models supported by the MiniMax music generation API.
MUSIC_GENERATION_MODELS = ["music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"]
MUSIC_COVER_MODELS = ["music-cover", "music-cover-free"]
MUSIC_MODELS = MUSIC_GENERATION_MODELS + MUSIC_COVER_MODELS
MUSIC_OUTPUT_FORMATS = ["url", "hex"]
MUSIC_AUDIO_FORMATS = ["mp3", "wav", "pcm"]

# Global and CN regional endpoints for the music generation API.
MUSIC_GENERATION_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}


class MiniMaxMusicProvider:
    """MiniMax music generation provider implementation.

    Wires the ``/v1/music_generation`` endpoint for both the global
    (api.minimax.io) and CN (api.minimaxi.com) regions. The region is inferred
    from the configured base URL. Generation models (``music-3.0``,
    ``music-2.6`` and their free variants) create music from a prompt and
    lyrics, while cover models (``music-cover`` and ``music-cover-free``) create
    a cover from a reference audio. Output can be returned as a ``url`` (valid
    for 24 hours) or as ``hex``-encoded audio.
    """

    def __init__(self):
        config = config_service.app_config.get("minimax", {})
        self.api_key = str(config.get("api_key", ""))
        self.base_url = str(config.get("url", "https://api.minimax.io")).rstrip("/")
        self.model_name = str(config.get("model_name", "music-3.0"))

        if not self.api_key:
            raise ValueError("MiniMax API key is not configured")
        if not self.base_url:
            raise ValueError("MiniMax URL is not configured")

    @property
    def region(self) -> str:
        """Region inferred from the configured base URL."""
        return "cn_zh" if "minimaxi.com" in self.base_url else "global_en"

    def _build_api_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/music_generation"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/music_generation"
        return f"{base_url}/v1/music_generation"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_payload(
        self,
        model: str,
        prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
        stream: bool = False,
        output_format: str = "url",
        audio_setting: Optional[Dict[str, Any]] = None,
        lyrics_optimizer: bool = False,
        is_instrumental: bool = False,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
        cover_feature_id: Optional[str] = None,
        aigc_watermark: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model}

        if prompt:
            payload["prompt"] = prompt
        if lyrics:
            payload["lyrics"] = lyrics
        if stream:
            payload["stream"] = True
        if output_format:
            payload["output_format"] = output_format
        if audio_setting:
            payload["audio_setting"] = audio_setting
        if lyrics_optimizer:
            payload["lyrics_optimizer"] = True
        if is_instrumental:
            payload["is_instrumental"] = True
        if audio_url:
            payload["audio_url"] = audio_url
        if audio_base64:
            payload["audio_base64"] = audio_base64
        if cover_feature_id:
            payload["cover_feature_id"] = cover_feature_id

        # ``aigc_watermark`` is only supported on the CN region.
        if self.region == "cn_zh" and aigc_watermark is not None:
            payload["aigc_watermark"] = aigc_watermark

        return payload

    def _validate_request(
        self,
        model: str,
        output_format: str,
        stream: bool,
        is_instrumental: bool,
        lyrics_optimizer: bool,
        prompt: Optional[str],
        lyrics: Optional[str],
        audio_setting: Optional[Dict[str, Any]],
        audio_url: Optional[str],
        audio_base64: Optional[str],
        cover_feature_id: Optional[str],
    ) -> None:
        if model not in MUSIC_MODELS:
            raise ValueError(
                f"Unsupported MiniMax music model: {model}. "
                f"Supported models are {', '.join(MUSIC_MODELS)}."
            )
        if output_format not in MUSIC_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format: {output_format}. "
                f"Allowed values are {', '.join(MUSIC_OUTPUT_FORMATS)}."
            )
        if stream and output_format != "hex":
            raise ValueError("Streaming music output only supports the hex format.")
        if audio_setting:
            audio_format = audio_setting.get("format")
            if audio_format and audio_format not in MUSIC_AUDIO_FORMATS:
                raise ValueError(
                    f"Unsupported audio format: {audio_format}. "
                    f"Allowed values are {', '.join(MUSIC_AUDIO_FORMATS)}."
                )

        if model in MUSIC_COVER_MODELS:
            reference_count = sum(
                bool(value) for value in (audio_url, audio_base64, cover_feature_id)
            )
            if reference_count != 1:
                raise ValueError(
                    "Music cover generation requires exactly one of audio_url, "
                    "audio_base64, or cover_feature_id."
                )
            if not prompt:
                raise ValueError("Music cover generation requires a prompt.")
            if cover_feature_id and not lyrics:
                raise ValueError(
                    "Music cover generation with cover_feature_id requires lyrics."
                )
            return

        if audio_url or audio_base64 or cover_feature_id:
            raise ValueError(
                "Reference audio inputs are only supported by music cover models."
            )
        if is_instrumental and not prompt:
            raise ValueError("Instrumental music generation requires a prompt.")
        if not is_instrumental and not lyrics:
            if lyrics_optimizer and prompt:
                return
            raise ValueError(
                "Non-instrumental music generation requires lyrics or "
                "lyrics_optimizer=True with a prompt."
            )

    def _extract_error_message(self, payload: Dict[str, Any], status: int) -> str:
        if isinstance(payload, dict):
            base_resp = payload.get("base_resp")
            if isinstance(base_resp, dict):
                message = base_resp.get("status_msg") or base_resp.get("message")
                if isinstance(message, str) and message.strip():
                    return message
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message
        return f"HTTP {status}"

    async def _request_json(
        self, session: Any, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        headers = self._build_headers()
        async with session.post(
            self._build_api_url(), headers=headers, json=payload
        ) as response:
            try:
                data = await response.json()
            except Exception:
                data = {"message": await response.text()}

            if response.status >= 400:
                if not isinstance(data, dict):
                    raise Exception(f"HTTP {response.status}")
                raise Exception(self._extract_error_message(data, response.status))

            if not isinstance(data, dict):
                raise Exception("MiniMax music API returned an invalid response")

            return data

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """Parse a MiniMax music generation response into an audio string.

        Success is signalled by ``base_resp.status_code == 0``. The audio is
        returned under ``data.audio`` once ``data.status`` is ``2`` (completed).
        """
        base_resp = response.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code != 0:
            message = base_resp.get("status_msg") or base_resp.get("message")
            raise Exception(
                f"MiniMax music generation failed"
                f" (base_resp.status_code={status_code}): {message}"
            )

        data = response.get("data") or {}
        status = data.get("status")
        if status == 2:
            audio = data.get("audio")
            if audio:
                return str(audio)
            raise Exception("No audio returned in successful music response")
        if status == 1:
            raise Exception(
                "MiniMax music generation is still in progress; "
                "no query endpoint is available for this task."
            )

        raise Exception(
            f"MiniMax music generation returned an unknown data.status: {status}"
        )

    async def generate(
        self,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
        stream: bool = False,
        output_format: str = "url",
        audio_setting: Optional[Dict[str, Any]] = None,
        lyrics_optimizer: bool = False,
        is_instrumental: bool = False,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
        cover_feature_id: Optional[str] = None,
        aigc_watermark: Optional[bool] = None,
    ) -> str:
        """Generate music and return the audio as a url or hex string."""
        try:
            selected_model = model or self.model_name
            self._validate_request(
                model=selected_model,
                output_format=output_format,
                stream=stream,
                is_instrumental=is_instrumental,
                lyrics_optimizer=lyrics_optimizer,
                prompt=prompt,
                lyrics=lyrics,
                audio_setting=audio_setting,
                audio_url=audio_url,
                audio_base64=audio_base64,
                cover_feature_id=cover_feature_id,
            )

            payload = self._build_request_payload(
                model=selected_model,
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

            async with HttpClient.create_aiohttp() as session:
                response = await self._request_json(session, payload)

            return self._parse_response(response)
        except Exception as e:
            print(f"🎵 Error generating music with MiniMax: {str(e)}")
            traceback.print_exc()
            raise e


__all__ = ["MiniMaxMusicProvider"]
