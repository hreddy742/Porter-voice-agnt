# ============================================================
# test_voice.py — standalone Cartesia TTS smoke test
# Verifies the API key + voice ID work and audio bytes come back.
# ============================================================

import asyncio
from dotenv import load_dotenv
from livekit.agents.utils import http_context
from livekit.plugins import cartesia

load_dotenv()

VOICE_ID = "f039066f-cdb7-45ed-b51d-1034ae2f04a0"  # same voice used in agent.py


async def main():
    async with http_context.open():
        tts = cartesia.TTS(voice=VOICE_ID)
        stream = tts.synthesize("Hi, this is a Cartesia text to speech test for Porter Capital.")

        total_bytes = 0
        async for frame in stream:
            total_bytes += len(frame.frame.data)

    print(f"Received {total_bytes} bytes of audio.")
    if total_bytes > 0:
        print("Cartesia TTS is working.")
    else:
        print("Cartesia TTS returned no audio.")


if __name__ == "__main__":
    asyncio.run(main())
