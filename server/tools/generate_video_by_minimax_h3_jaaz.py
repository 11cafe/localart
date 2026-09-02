from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool  # type: ignore
from pydantic import BaseModel, Field

from .utils.image_utils import process_input_image
from .video_generation import generate_video_with_provider


class GenerateVideoByMiniMaxH3InputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for video generation. Describe the scene clearly."
    )
    resolution: str = Field(
        default="2K",
        description="Optional. The only supported resolution is 2K.",
    )
    duration: int = Field(
        default=5,
        description="Optional. Duration in seconds. Allowed values are 4 through 15.",
    )
    ratio: str = Field(
        default="16:9",
        description="Optional. The aspect ratio. Allowed values are adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.",
    )
    input_images: list[str] | None = Field(
        default=None,
        description="Optional. Images to use as frame or reference inputs. One image uses the first frame role; two images use first and last frame roles.",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional. Callback URL for task updates.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "generate_video_by_minimax_h3_jaaz",
    description="Generate videos using MiniMax H3 with 2K output, 4-15 second duration, and text or image inputs.",
    args_schema=GenerateVideoByMiniMaxH3InputSchema,
)
async def generate_video_by_minimax_h3_jaaz(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    resolution: str = "2K",
    duration: int = 5,
    ratio: str = "16:9",
    input_images: list[str] | None = None,
    callback_url: str | None = None,
) -> str:
    processed_input_images = None
    if input_images and len(input_images) > 0:
        processed_input_images = []
        for image_name in input_images:
            processed_image = await process_input_image(image_name)
            if not processed_image:
                raise ValueError(
                    f"Failed to process input image: {image_name}. Please check if the image exists and is valid."
                )
            processed_input_images.append(processed_image)

    return await generate_video_with_provider(
        prompt=prompt,
        resolution=resolution,
        duration=duration,
        aspect_ratio=ratio,
        model="MiniMax-H3",
        tool_call_id=tool_call_id,
        config=config,
        input_images=processed_input_images,
        callback_url=callback_url,
    )


__all__ = ["generate_video_by_minimax_h3_jaaz"]
