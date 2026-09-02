import asyncio

import pytest

from services.config_service import config_service
from tools.video_providers.minimax_provider import MiniMaxVideoProvider


def make_provider(monkeypatch, url: str = "https://api.minimax.io") -> MiniMaxVideoProvider:
    monkeypatch.setitem(
        config_service.app_config,
        "minimax",
        {
            "api_key": "test-key",
            "url": url,
            "model_name": "MiniMax-H3",
        },
    )
    return MiniMaxVideoProvider()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.minimax.io", "https://api.minimax.io/v2/video_generation"),
        ("https://api.minimax.io/v2", "https://api.minimax.io/v2/video_generation"),
        (
            "https://api.minimax.io/v2/video_generation",
            "https://api.minimax.io/v2/video_generation",
        ),
        (
            "https://api.minimaxi.com",
            "https://api.minimaxi.com/v2/video_generation",
        ),
    ],
)
def test_minimax_builds_video_endpoint(monkeypatch, url: str, expected: str):
    provider = make_provider(monkeypatch, url)
    assert provider._build_api_url() == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/query/video_generation/task-123",
            "https://api.minimax.io/v2/query/video_generation/task-123",
        ),
        (
            "/query/video_generation",
            "https://api.minimax.io/v2/query/video_generation",
        ),
        (
            "/video_generation/task-123",
            "https://api.minimax.io/v2/video_generation/task-123",
        ),
    ],
)
def test_minimax_builds_operation_endpoints(monkeypatch, path: str, expected: str):
    provider = make_provider(monkeypatch)
    assert provider._build_api_url(path) == expected


def test_minimax_uses_v2_operation_paths(monkeypatch):
    provider = make_provider(monkeypatch)
    requests = []

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    async def fake_request_json(
        session, method, path, headers, payload=None, params=None
    ):
        requests.append((method, path, payload, params))
        if path.endswith("task-poll"):
            return {
                "task": {
                    "status": "succeeded",
                    "content": {"url": "https://example.com/video.mp4"},
                }
            }
        return {"task_id": "task-123"}

    monkeypatch.setattr(provider, "_request_json", fake_request_json)
    monkeypatch.setattr(
        "tools.video_providers.minimax_provider.HttpClient.create_aiohttp",
        lambda: SessionContext(),
    )

    async def exercise_operations():
        await provider.create_video_generation_task(
            content=[{"type": "text", "text": "A cinematic scene"}],
            resolution="2K",
            duration=5,
            ratio="16:9",
        )
        await provider.query_video_generation_task("task-123")
        await provider.list_video_generation_tasks()
        await provider.delete_video_generation_task("task-123")
        assert (
            await provider._poll_for_task_completion("task-poll")
            == "https://example.com/video.mp4"
        )

    asyncio.run(exercise_operations())

    assert [(method, path) for method, path, _, _ in requests] == [
        ("POST", "/video_generation"),
        ("GET", "/query/video_generation/task-123"),
        ("GET", "/query/video_generation"),
        ("DELETE", "/video_generation/task-123"),
        ("GET", "/query/video_generation/task-poll"),
    ]


def test_minimax_builds_frame_roles(monkeypatch):
    provider = make_provider(monkeypatch)
    content = provider._build_content(
        "A cinematic scene",
        input_images=["data:image/png;base64,one", "data:image/png;base64,two"],
    )

    assert content[0] == {"type": "text", "text": "A cinematic scene"}
    assert content[1]["role"] == "first_frame"
    assert content[2]["role"] == "last_frame"


def test_minimax_builds_reference_roles(monkeypatch):
    provider = make_provider(monkeypatch)
    content = provider._build_content(
        "A cinematic scene",
        reference_images=["data:image/png;base64,one"],
        reference_videos=["https://example.com/ref.mp4"],
        reference_audios=["https://example.com/ref.mp3"],
    )

    roles = [item.get("role") for item in content[1:]]
    assert roles == ["reference_image", "reference_video", "reference_audio"]


def test_minimax_rejects_mixed_roles(monkeypatch):
    provider = make_provider(monkeypatch)

    with pytest.raises(ValueError, match="mutually exclusive"):
        provider._build_content(
            "A cinematic scene",
            input_images=["data:image/png;base64,one"],
            reference_images=["data:image/png;base64,two"],
        )


@pytest.mark.parametrize(
    ("resolution", "duration", "ratio", "message"),
    [
        ("1080p", 5, "16:9", "2K resolution"),
        ("2K", 3, "16:9", "between 4 and 15"),
        ("2K", 5, "adaptive", "concrete ratio"),
    ],
)
def test_minimax_validates_request(monkeypatch, resolution, duration, ratio, message):
    provider = make_provider(monkeypatch)
    content = provider._build_content("A cinematic scene")

    with pytest.raises(ValueError, match=message):
        provider._validate_request(resolution, duration, ratio, content)
