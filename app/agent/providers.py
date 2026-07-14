"""LLM and TTS provider selection for Aiva."""

import os

from dotenv import load_dotenv
from livekit.agents import inference
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import anthropic, cartesia, elevenlabs, groq
from livekit.plugins import openai as openai_plugin

load_dotenv()


# ============================================================
# LLM SELECTION
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# How long Ollama keeps qwen3:4b resident in memory after a request. Default
# (a few minutes) means the model unloads between calls during normal
# testing/campaign pacing, so the very failover this warm-up exists for
# would still hit a cold load. 30m comfortably spans a testing session.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# xAI Grok via LiveKit Inference (billed through LiveKit Cloud; no XAI_API_KEY).
# Override with GROK_MODEL if you want a different Grok SKU.
GROK_MODEL = os.getenv("GROK_MODEL", "xai/grok-4-1-fast-non-reasoning")


def _build_llm(provider: str):
    if provider in ("grok", "xai"):
        # Prefer LiveKit Inference when no XAI_API_KEY is set. If XAI_API_KEY
        # is present, go direct to xAI via the OpenAI-compatible helper.
        if os.getenv("XAI_API_KEY"):
            return openai_plugin.LLM.with_x_ai(
                model=os.getenv("GROK_MODEL", "grok-3-fast").removeprefix("xai/"),
                api_key=os.getenv("XAI_API_KEY"),
            )
        return inference.LLM(
            model=GROK_MODEL,
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
    if provider == "groq":
        return groq.LLM(model="llama-3.3-70b-versatile", timeout=10.0)
    elif provider == "claude":
        return anthropic.LLM(model="claude-haiku-4-5")
    elif provider == "gpt":
        return openai_plugin.LLM(model="gpt-4o-mini")
    elif provider == "qwen":
        # qwen3:4b defaults to an internal "thinking" mode that generates a
        # long hidden reasoning chain before any tool call/content — this
        # measured ~45s to produce a single tool call locally, which reads
        # as total dead silence in a live call. enable_thinking=False (via
        # Ollama's OpenAI-compatible chat_template_kwargs pass-through)
        # drops that to ~4-5s. Confirmed empirically before relying on it.
        return openai_plugin.LLM(
            model="qwen3:4b",
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
        )
    elif provider == "mistral-local":
        # Stand-in for mistral-small until it's actually pulled in Ollama.
        return openai_plugin.LLM.with_ollama(model="gemma4:latest", base_url=OLLAMA_BASE_URL)
    elif provider == "gptoss":
        # Stand-in for gpt-oss:20b until it's actually pulled in Ollama.
        return openai_plugin.LLM.with_ollama(model="llama3.1:latest", base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "grok").lower()
    print(f"=== Testing LLM: {provider.upper()} ===")

    primary = _build_llm(provider)

    # Optional failover (e.g. Groq cloud -> local qwen). Empty
    # LLM_FAILOVER_PROVIDER disables this path.
    failover_provider = os.getenv("LLM_FAILOVER_PROVIDER", "").lower()
    if failover_provider and failover_provider != provider:
        try:
            fallback = _build_llm(failover_provider)
        except Exception as e:
            print(f"WARNING: could not build LLM_FAILOVER_PROVIDER={failover_provider} ({e}); continuing without failover")
            return primary
        return FallbackAdapter(llm=[primary, fallback])

    return primary


# ============================================================
# TTS SELECTION
# ============================================================
#
# Neither Kokoro nor Chatterbox-Turbo has an official livekit-plugins
# package. Both are wired in via livekit-plugins-openai's TTS class
# pointed at a local OpenAI-compatible server (`/v1/audio/speech`),
# same pattern as get_llm() uses openai_plugin.LLM.with_ollama() for
# local models:
#   - Kokoro: run a Kokoro-FastAPI server (OpenAI-compatible) locally.
#   - Chatterbox-Turbo: the chatterbox-tts pip package is library-only
#     (no bundled server) — run it behind a community OpenAI-compatible
#     wrapper such as devnen/Chatterbox-TTS-Server.

KOKORO_BASE_URL     = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
CHATTERBOX_BASE_URL = os.getenv("CHATTERBOX_BASE_URL", "http://localhost:8880/v1")


def get_tts():
    provider = os.getenv("TTS_PROVIDER", "cartesia").lower()
    print(f"=== Testing TTS: {provider.upper()} ===")

    if provider == "cartesia":
        return cartesia.TTS(
            voice="a33f7a4c-100f-41cf-a1fd-5822e8fc253f",
            speed=1.0,
        )
    elif provider == "kokoro":
        return openai_plugin.TTS(
            model="kokoro",
            voice="af_bella",
            base_url=KOKORO_BASE_URL,
            api_key="not-needed",
        )
    elif provider == "chatterbox":
        return openai_plugin.TTS(
            model="chatterbox",
            voice="default",
            base_url=CHATTERBOX_BASE_URL,
            api_key="not-needed",
        )
    elif provider == "elevenlabs":
        return elevenlabs.TTS(
            voice_id="56bWURjYFHyYyVf490Dp",
            model="eleven_turbo_v2_5",
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {provider}")
