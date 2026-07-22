"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class Settings:
    agent_name: str
    agent_version: str
    livekit_url: str
    livekit_inference_api_key: str
    livekit_inference_api_secret: str
    test_mode: bool

    @classmethod
    def from_env(cls, *, allow_livekit_cloud: bool = False) -> "Settings":
        livekit_url = os.getenv("LIVEKIT_URL", "")
        hostname = (urlparse(livekit_url).hostname or "").lower()
        if not livekit_url:
            raise ValueError("LIVEKIT_URL is required")
        is_livekit_cloud = hostname == "livekit.cloud" or hostname.endswith(
            ".livekit.cloud"
        )
        if is_livekit_cloud and not allow_livekit_cloud:
            raise ValueError("LiveKit Cloud is disabled; configure a self-hosted server")
        inference_api_key = os.getenv("LIVEKIT_INFERENCE_API_KEY") or os.getenv(
            "LIVEKIT_API_KEY", ""
        )
        inference_api_secret = os.getenv("LIVEKIT_INFERENCE_API_SECRET") or os.getenv(
            "LIVEKIT_API_SECRET", ""
        )
        if not inference_api_key or not inference_api_secret:
            raise ValueError(
                "LiveKit Inference credentials are required; configure either "
                "LIVEKIT_INFERENCE_API_KEY and LIVEKIT_INFERENCE_API_SECRET or "
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )
        return cls(
            agent_name=os.getenv("AGENT_NAME", "codex-agent"),
            agent_version=os.getenv("AGENT_VERSION", "v2"),
            livekit_url=livekit_url,
            livekit_inference_api_key=inference_api_key,
            livekit_inference_api_secret=inference_api_secret,
            test_mode=os.getenv("TEST_MODE", "false").lower() == "true",
        )
