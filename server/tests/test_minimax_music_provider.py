import pytest
from services.config_service import config_service
from tools.music_providers.minimax_provider import (
    MUSIC_COVER_MODELS,
    MUSIC_GENERATION_MODELS,
    MiniMaxMusicProvider,
)


def make_provider(
    monkeypatch, url: str = "https://api.minimax.io"
) -> MiniMaxMusicProvider:
    monkeypatch.setitem(
        config_service.app_config,
        "minimax",
        {
            "api_key": "test-key",
            "url": url,
            "model_name": "music-3.0",
        },
    )
    return MiniMaxMusicProvider()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.minimax.io", "https://api.minimax.io/v1/music_generation"),
        ("https://api.minimax.io/v1", "https://api.minimax.io/v1/music_generation"),
        (
            "https://api.minimax.io/v1/music_generation",
            "https://api.minimax.io/v1/music_generation",
        ),
        (
            "https://api.minimaxi.com",
            "https://api.minimaxi.com/v1/music_generation",
        ),
    ],
)
def test_minimax_builds_music_endpoint(monkeypatch, url: str, expected: str):
    provider = make_provider(monkeypatch, url)
    assert provider._build_api_url() == expected


def test_minimax_detects_region(monkeypatch):
    assert make_provider(monkeypatch, "https://api.minimax.io").region == "global_en"
    assert make_provider(monkeypatch, "https://api.minimaxi.com").region == "cn_zh"


def test_minimax_supports_generation_and_cover_models():
    assert MUSIC_GENERATION_MODELS == [
        "music-3.0",
        "music-2.6",
        "music-3.0-free",
        "music-2.6-free",
    ]
    assert MUSIC_COVER_MODELS == ["music-cover", "music-cover-free"]


def test_minimax_builds_bearer_headers(monkeypatch):
    provider = make_provider(monkeypatch)
    assert provider._build_headers() == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_minimax_builds_generation_payload(monkeypatch):
    provider = make_provider(monkeypatch)
    payload = provider._build_request_payload(
        model="music-3.0",
        prompt="Pop, melancholic, perfect for a rainy night",
        lyrics="[Chorus]\nHello world",
        output_format="url",
        audio_setting={"format": "mp3", "sample_rate": 44100, "bitrate": 128000},
        lyrics_optimizer=True,
    )

    assert payload["model"] == "music-3.0"
    assert payload["prompt"].startswith("Pop")
    assert "[Chorus]" in payload["lyrics"]
    assert payload["output_format"] == "url"
    assert payload["audio_setting"]["format"] == "mp3"
    assert payload["lyrics_optimizer"] is True
    assert "stream" not in payload


def test_minimax_builds_cover_payload(monkeypatch):
    provider = make_provider(monkeypatch)
    payload = provider._build_request_payload(
        model="music-cover",
        prompt="A jazz remake",
        audio_url="https://example.com/song.mp3",
        stream=True,
        output_format="hex",
    )

    assert payload["model"] == "music-cover"
    assert payload["audio_url"] == "https://example.com/song.mp3"
    assert payload["stream"] is True
    assert payload["output_format"] == "hex"


def test_minimax_cn_adds_watermark(monkeypatch):
    provider = make_provider(monkeypatch, "https://api.minimaxi.com")
    payload = provider._build_request_payload(
        model="music-3.0",
        prompt="Pop",
        aigc_watermark=True,
    )

    assert payload["aigc_watermark"] is True


def test_minimax_global_omits_watermark(monkeypatch):
    provider = make_provider(monkeypatch, "https://api.minimax.io")
    payload = provider._build_request_payload(
        model="music-3.0",
        prompt="Pop",
        aigc_watermark=True,
    )

    assert "aigc_watermark" not in payload


def test_minimax_validates_request(monkeypatch):
    provider = make_provider(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported MiniMax music model"):
        provider._validate_request(
            model="unknown-model",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="Pop",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )

    with pytest.raises(ValueError, match="output format"):
        provider._validate_request(
            model="music-3.0",
            output_format="wav",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="Pop",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )

    with pytest.raises(ValueError, match="Streaming"):
        provider._validate_request(
            model="music-3.0",
            output_format="url",
            stream=True,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="Pop",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )

    with pytest.raises(ValueError, match="cover generation requires"):
        provider._validate_request(
            model="music-cover",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="A jazz remake",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )


def test_minimax_validates_conditional_inputs(monkeypatch):
    provider = make_provider(monkeypatch)

    with pytest.raises(ValueError, match="requires a prompt"):
        provider._validate_request(
            model="music-3.0",
            output_format="url",
            stream=False,
            is_instrumental=True,
            lyrics_optimizer=False,
            prompt=None,
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )

    with pytest.raises(ValueError, match="requires lyrics"):
        provider._validate_request(
            model="music-3.0",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="Pop",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )

    provider._validate_request(
        model="music-3.0",
        output_format="url",
        stream=False,
        is_instrumental=False,
        lyrics_optimizer=True,
        prompt="Pop",
        lyrics=None,
        audio_setting=None,
        audio_url=None,
        audio_base64=None,
        cover_feature_id=None,
    )

    with pytest.raises(ValueError, match="exactly one"):
        provider._validate_request(
            model="music-cover",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="A jazz remake",
            lyrics=None,
            audio_setting=None,
            audio_url="https://example.com/song.mp3",
            audio_base64="dGVzdA==",
            cover_feature_id=None,
        )

    with pytest.raises(ValueError, match="cover_feature_id requires lyrics"):
        provider._validate_request(
            model="music-cover",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="A jazz remake",
            lyrics=None,
            audio_setting=None,
            audio_url=None,
            audio_base64=None,
            cover_feature_id="feature-id",
        )

    with pytest.raises(ValueError, match="Unsupported audio format"):
        provider._validate_request(
            model="music-3.0",
            output_format="url",
            stream=False,
            is_instrumental=False,
            lyrics_optimizer=False,
            prompt="Pop",
            lyrics="[Chorus]\nHello world",
            audio_setting={"format": "flac"},
            audio_url=None,
            audio_base64=None,
            cover_feature_id=None,
        )


def test_minimax_parses_completed_url(monkeypatch):
    provider = make_provider(monkeypatch)
    response = {
        "data": {"status": 2, "audio": "https://example.com/music.mp3"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }

    assert provider._parse_response(response) == "https://example.com/music.mp3"


def test_minimax_parses_completed_hex(monkeypatch):
    provider = make_provider(monkeypatch)
    response = {
        "data": {"status": 2, "audio": "68656c6c6f"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }

    assert provider._parse_response(response) == "68656c6c6f"


def test_minimax_rejects_in_progress(monkeypatch):
    provider = make_provider(monkeypatch)
    response = {
        "data": {"status": 1},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }

    with pytest.raises(Exception, match="still in progress"):
        provider._parse_response(response)


def test_minimax_rejects_error_status_code(monkeypatch):
    provider = make_provider(monkeypatch)
    response = {
        "data": {},
        "base_resp": {"status_code": 2013, "status_msg": "invalid parameters"},
    }

    with pytest.raises(Exception, match="status_code=2013"):
        provider._parse_response(response)


def test_minimax_extracts_error_message(monkeypatch):
    provider = make_provider(monkeypatch)
    payload = {"base_resp": {"status_code": 1004, "status_msg": "auth failed"}}
    assert provider._extract_error_message(payload, 401) == "auth failed"
    assert provider._extract_error_message({}, 500) == "HTTP 500"
