import os
from typing import Any, Optional

from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient

from .image_base_provider import ImageProviderBase
from ..utils.image_utils import generate_image_id, get_image_info_and_save


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
        api_url = str(config.get("url", ""))

        if not api_key:
            raise ValueError("MiniMax API key is not configured")
        if not api_url:
            raise ValueError("MiniMax API URL is not configured")
        if input_images:
            raise ValueError("MiniMax text-to-image generation does not accept input images")

        payload = {
            "model": model.replace("minimax/", ""),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "response_format": "url",
            "n": kwargs.get("num_images", 1),
            "prompt_optimizer": kwargs.get("prompt_optimizer", True),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with HttpClient.create_aiohttp() as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                response_json = await response.json()

        base_status = response_json.get("base_resp", {}).get("status_code")
        image_urls = response_json.get("data", {}).get("image_urls", [])
        if response.status != 200 or base_status not in (None, 0) or not image_urls:
            raise RuntimeError(f"MiniMax image generation failed: {response_json}")

        image_id = generate_image_id()
        mime_type, width, height, extension = await get_image_info_and_save(
            image_urls[0], os.path.join(FILES_DIR, image_id)
        )
        if mime_type is None:
            raise RuntimeError("Failed to determine generated image MIME type")

        return mime_type, width, height, f"{image_id}.{extension}"
