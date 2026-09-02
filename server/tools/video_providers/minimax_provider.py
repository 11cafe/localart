import asyncio
import traceback
from typing import Any, Dict, List, Optional

from services.config_service import config_service
from utils.http_client import HttpClient

from .video_base_provider import VideoProviderBase


class MiniMaxVideoProvider(VideoProviderBase, provider_name="minimax"):
    """MiniMax H3 video generation provider implementation"""

    def __init__(self):
        config = config_service.app_config.get("minimax", {})
        self.api_key = str(config.get("api_key", ""))
        self.base_url = str(config.get("url", "https://api.minimax.io")).rstrip("/")
        self.model_name = str(config.get("model_name", "MiniMax-H3"))

        if not self.api_key:
            raise ValueError("MiniMax API key is not configured")
        if not self.base_url:
            raise ValueError("MiniMax URL is not configured")

    def _build_api_url(self, path: str = "/video_generation") -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v2/video_generation"):
            api_root = base_url.removesuffix("/video_generation")
        elif base_url.endswith("/v2"):
            api_root = base_url
        else:
            api_root = f"{base_url}/v2"

        return f"{api_root}/{path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _clean_values(values: Optional[List[str]]) -> List[str]:
        return [value.strip() for value in values or [] if value and value.strip()]

    def _build_content(
        self,
        prompt: str,
        input_images: Optional[List[str]] = None,
        reference_images: Optional[List[str]] = None,
        reference_videos: Optional[List[str]] = None,
        reference_audios: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        text = prompt.strip()
        if not text:
            raise ValueError("Prompt is required")
        if len(text) > 7000:
            raise ValueError("Prompt must be 7000 characters or fewer")

        images = self._clean_values(input_images)
        refs = self._clean_values(reference_images)
        videos = self._clean_values(reference_videos)
        audios = self._clean_values(reference_audios)

        if audios and not (images or refs or videos):
            raise ValueError("Reference audio requires a reference image or video")

        if (images and (refs or videos or audios)) or (
            len(images) == 2 and (refs or videos or audios)
        ):
            raise ValueError("Frame roles and reference roles are mutually exclusive")

        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

        if images and not (refs or videos or audios):
            if len(images) == 1:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": images[0]},
                        "role": "first_frame",
                    }
                )
            elif len(images) == 2:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": images[0]},
                        "role": "first_frame",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": images[1]},
                        "role": "last_frame",
                    }
                )
            else:
                for image_url in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                            "role": "reference_image",
                        }
                    )
            return content

        for image_url in images + refs:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "role": "reference_image",
                }
            )

        for video_url in videos:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "role": "reference_video",
                }
            )

        for audio_url in audios:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_url},
                    "role": "reference_audio",
                }
            )

        return content

    def _validate_request(
        self,
        resolution: str,
        duration: int,
        ratio: str,
        content: List[Dict[str, Any]],
    ) -> None:
        if resolution != "2K":
            raise ValueError("MiniMax H3 only supports 2K resolution")
        if not isinstance(duration, int) or duration < 4 or duration > 15:
            raise ValueError("MiniMax H3 duration must be an integer between 4 and 15")
        if len(content) == 1 and ratio == "adaptive":
            raise ValueError("Text-to-video requests require a concrete ratio")

    @staticmethod
    def _extract_error_message(payload: Dict[str, Any], status: int) -> str:
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = value.get("message") or value.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested
        base_resp = payload.get("base_resp")
        if isinstance(base_resp, dict):
            message = base_resp.get("status_msg") or base_resp.get("message")
            if isinstance(message, str) and message.strip():
                return message
        return f"HTTP {status}"

    @staticmethod
    def _extract_task(response: Dict[str, Any]) -> Dict[str, Any]:
        task = response.get("task")
        if isinstance(task, dict):
            return task
        return response

    @staticmethod
    def _normalize_status(status: Any) -> str:
        return str(status or "").strip().lower()

    @staticmethod
    def _extract_video_url(task: Dict[str, Any]) -> Optional[str]:
        content = task.get("content")
        if isinstance(content, dict):
            url = content.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    url = item.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()
        return None

    async def _request_json(
        self,
        session: Any,
        method: str,
        path: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with session.request(
            method,
            self._build_api_url(path),
            headers=headers,
            json=payload,
            params=params,
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
                raise Exception("MiniMax video API returned an invalid response")

            return data

    async def create_video_generation_task(
        self,
        content: List[Dict[str, Any]],
        resolution: str,
        duration: int,
        ratio: str,
        callback_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        headers = self._build_headers()
        payload: Dict[str, Any] = {
            "model": model or self.model_name,
            "content": content,
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
        }
        if callback_url:
            payload["callback_url"] = callback_url

        async with HttpClient.create_aiohttp() as session:
            response = await self._request_json(
                session,
                "POST",
                "/video_generation",
                headers,
                payload=payload,
            )

        task_id = str(response.get("task_id", "")).strip()
        if not task_id:
            raise Exception("MiniMax video generation task creation failed")

        return task_id

    async def query_video_generation_task(self, task_id: str) -> Dict[str, Any]:
        headers = self._build_headers()
        async with HttpClient.create_aiohttp() as session:
            return await self._request_json(
                session,
                "GET",
                f"/query/video_generation/{task_id}",
                headers,
            )

    async def list_video_generation_tasks(
        self,
        page_num: int = 1,
        page_size: int = 20,
        filter_status: Optional[str] = None,
        filter_task_ids: Optional[List[str]] = None,
        filter_model: Optional[str] = None,
        filter_task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        headers = self._build_headers()
        params: Dict[str, Any] = {
            "page_num": page_num,
            "page_size": page_size,
        }
        if filter_status:
            params["filter.status"] = filter_status
        if filter_task_ids:
            params["filter.task_ids"] = filter_task_ids
        if filter_model:
            params["filter.model"] = filter_model
        if filter_task_type:
            params["filter.task_type"] = filter_task_type

        async with HttpClient.create_aiohttp() as session:
            return await self._request_json(
                session,
                "GET",
                "/query/video_generation",
                headers,
                params=params,
            )

    async def delete_video_generation_task(self, task_id: str) -> Dict[str, Any]:
        headers = self._build_headers()
        async with HttpClient.create_aiohttp() as session:
            return await self._request_json(
                session,
                "DELETE",
                f"/video_generation/{task_id}",
                headers,
            )

    async def _poll_for_task_completion(self, task_id: str) -> str:
        headers = self._build_headers()
        async with HttpClient.create_aiohttp() as session:
            while True:
                response = await self._request_json(
                    session,
                    "GET",
                    f"/query/video_generation/{task_id}",
                    headers,
                )
                task = self._extract_task(response)
                status = self._normalize_status(task.get("status"))

                if status in {"succeeded", "success", "completed", "done"}:
                    video_url = self._extract_video_url(task)
                    if video_url:
                        return video_url
                    raise Exception("No video URL found in successful response")

                if status in {"failed", "cancelled", "canceled", "error", "expired"}:
                    detail_error = task.get("error") or task.get("message")
                    if not detail_error:
                        detail_error = f"Task failed with status: {status}"
                    raise Exception(f"MiniMax video generation failed: {detail_error}")

                print(
                    f"🎥 Polling MiniMax H3 generation {task_id}, current status: {status} ..."
                )
                await asyncio.sleep(5)

    async def generate(
        self,
        prompt: str,
        model: str,
        resolution: str = "2K",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_images: Optional[List[str]] = None,
        camera_fixed: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            selected_model = model or self.model_name
            if selected_model != "MiniMax-H3":
                raise ValueError("MiniMax video provider only supports MiniMax-H3")

            reference_images = kwargs.pop("reference_images", None)
            reference_videos = kwargs.pop("reference_videos", None)
            reference_audios = kwargs.pop("reference_audios", None)
            callback_url = kwargs.pop("callback_url", None)
            ratio = str(kwargs.pop("ratio", aspect_ratio))

            content = self._build_content(
                prompt=prompt,
                input_images=input_images,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
            )
            self._validate_request(resolution, duration, ratio, content)

            task_id = await self.create_video_generation_task(
                content=content,
                resolution=resolution,
                duration=duration,
                ratio=ratio,
                callback_url=callback_url,
                model=selected_model,
            )

            video_url = await self._poll_for_task_completion(task_id)
            print(f"🎥 MiniMax H3 video generation completed, video URL: {video_url}")
            return video_url
        except Exception as e:
            print(f"🎥 Error generating video with MiniMax H3: {str(e)}")
            traceback.print_exc()
            raise e


__all__ = ["MiniMaxVideoProvider"]
