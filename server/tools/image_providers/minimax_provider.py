import os
from typing import Any, Optional

from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient

from ..utils.image_utils import generate_image_id, get_image_info_and_save
from .image_base_provider import ImageProviderBase


class MiniMaxImageProvider(ImageProviderBase):
    """MiniMax image generation provider implementation."""

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, str]:
        config = config_service.app_config.get("minimax", {})
        api_key = str(config.get("api_key", ""))
        api_url = str(
            config.get("url", "https://api.minimax.io/v1/image_generation")
        )

        if not api_key:
            raise ValueError("MiniMax API key is not configured")
        if not api_url:
            raise ValueError("MiniMax API URL is not configured")

        payload: dict[str, Any] = {
            "model": model.replace("minimax/", ""),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "response_format": kwargs.get("response_format", "url"),
            "n": kwargs.get("num_images", 1),
            "prompt_optimizer": kwargs.get("prompt_optimizer", False),
        }

        if input_images:
            payload["subject_reference"] = [
                {"type": "character", "image_file": image}
                for image in input_images
            ]

        for field in ("width", "height", "seed"):
            if field in kwargs and kwargs[field] is not None:
                payload[field] = kwargs[field]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with HttpClient.create_aiohttp() as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                response_json = await response.json()
                response_status = response.status

        base_status = response_json.get("base_resp", {}).get("status_code")
        if response_status != 200 or base_status not in (None, 0):
            raise RuntimeError(f"MiniMax image generation failed: {response_json}")

        data = response_json.get("data", {})
        response_format = payload["response_format"]
        if response_format == "base64":
            images = data.get("image_base64", [])
            is_b64 = True
        else:
            images = data.get("image_urls", [])
            is_b64 = False

        if not images:
            raise RuntimeError("MiniMax image generation returned no images")

        image_id = generate_image_id()
        mime_type, width, height, extension = await get_image_info_and_save(
            images[0],
            os.path.join(FILES_DIR, image_id),
            is_b64=is_b64,
        )
        if mime_type is None:
            raise RuntimeError("Failed to determine generated image MIME type")

        return mime_type, width, height, f"{image_id}.{extension}"


__all__ = ["MiniMaxImageProvider"]
