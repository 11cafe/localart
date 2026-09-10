from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool  # type: ignore
from pydantic import BaseModel, Field
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByMiniMaxInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation or image editing."
    )
    aspect_ratio: str = Field(
        default="1:1",
        description="Optional. Supported ratios include 1:1, 16:9, 4:3, 3:2, 2:3, 3:4, 9:16, and 21:9.",
    )
    model: str = Field(
        default="image-01",
        description="Optional. MiniMax image model, either image-01 or image-01-live.",
    )
    input_images: list[str] | None = Field(
        default=None,
        description="Optional. Reference image IDs for image-to-image generation.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "generate_image_by_minimax",
    description="Generate or edit an image with MiniMax using a text prompt and optional reference image.",
    args_schema=GenerateImageByMiniMaxInputSchema,
)
async def generate_image_by_minimax(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    model: str = "image-01",
    input_images: list[str] | None = None,
) -> str:
    ctx = config.get("configurable", {})
    return await generate_image_with_provider(
        canvas_id=ctx.get("canvas_id", ""),
        session_id=ctx.get("session_id", ""),
        provider="minimax",
        model=model,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        input_images=input_images,
    )


__all__ = ["generate_image_by_minimax"]
