"""Place a test call through a self-hosted LiveKit SIP deployment."""

import asyncio
import os
import time

from dotenv import load_dotenv
from livekit import api


load_dotenv()


async def main() -> None:
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
    destination = os.getenv("MY_PHONE_NUMBER")
    agent_name = os.getenv("AGENT_NAME", "codex-agent")

    missing = [
        name
        for name, value in (
            ("LIVEKIT_URL", url),
            ("LIVEKIT_API_KEY", key),
            ("LIVEKIT_API_SECRET", secret),
            ("SIP_OUTBOUND_TRUNK_ID", trunk_id),
            ("MY_PHONE_NUMBER", destination),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment values: {', '.join(missing)}")
    if ".livekit.cloud" in url.lower():
        raise RuntimeError("scripts.call supports self-hosted LiveKit only")

    room_name = f"porter-call-{int(time.time())}"
    livekit = api.LiveKitAPI(url=url, api_key=key, api_secret=secret)
    try:
        await livekit.room.create_room(api.CreateRoomRequest(name=room_name))
        await livekit.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=agent_name, room=room_name)
        )
        participant = await livekit.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=destination,
                room_name=room_name,
                participant_identity="prospect",
                participant_name="Prospect",
                wait_until_answered=True,
            )
        )
    finally:
        await livekit.aclose()

    print(
        f"Self-hosted LiveKit SIP call started: room={room_name} "
        f"participant={participant.participant_identity}"
    )


if __name__ == "__main__":
    asyncio.run(main())
