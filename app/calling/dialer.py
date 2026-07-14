# ============================================================
# dialer.py — Make Aiva call a real phone number
# Uses Twilio Programmable Voice (works on trial accounts)
# Usage: python call.py  (thin root wrapper)
# ============================================================

import os
import time
import asyncio
from dotenv import load_dotenv
from livekit import api
from twilio.rest import Client

load_dotenv()

# Settings from .env
LIVEKIT_URL    = os.getenv("LIVEKIT_URL")
LIVEKIT_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_SECRET = os.getenv("LIVEKIT_API_SECRET")
# Account SID must be AC.... An SK... value is an API Key SID, not an Account SID.
TWILIO_SID     = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
# Optional: when using a Twilio API Key (SK...) + secret instead of Auth Token.
TWILIO_API_KEY = os.getenv("TWILIO_API_KEY", "")
TWILIO_API_SECRET = os.getenv("TWILIO_API_SECRET", "")
TWILIO_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER")
MY_NUMBER      = os.getenv("MY_PHONE_NUMBER")

# LiveKit Cloud's Twilio Media Streams connector for your project
LIVEKIT_TWILIO_WS_URL = os.getenv(
    "LIVEKIT_TWILIO_WS_URL",
    "wss://0fsr7yjvwys.sip.livekit.cloud/twilio",
)


async def create_room_and_dispatch_aiva(room_name):
    """
    Creates a LiveKit room. Aiva's worker process (python agent.py)
    is running with automatic dispatch, so it joins this room on its
    own as soon as the room exists — no explicit dispatch call is needed.
    Aiva will wait in the room until the phone call connects.
    """
    lk = api.LiveKitAPI(
        url        = LIVEKIT_URL.replace("wss://", "https://"),
        api_key    = LIVEKIT_KEY,
        api_secret = LIVEKIT_SECRET,
    )

    # Create the room
    await lk.room.create_room(
        api.CreateRoomRequest(name=room_name)
    )
    print(f"Room created: {room_name}")

    # Generate a token for the phone participant
    # This token is what Twilio uses to join the room
    token = api.AccessToken(LIVEKIT_KEY, LIVEKIT_SECRET)
    token.with_identity("prospect")
    token.with_name("Prospect")
    token.with_grants(api.VideoGrants(room_join=True, room=room_name))
    prospect_token = token.to_jwt()

    await lk.aclose()
    return prospect_token


def _twilio_client() -> Client:
    """Build a Twilio REST client from .env credentials.

    Supported setups:
    1. Account SID (AC...) + Auth Token
    2. Account SID (AC...) + API Key SID (SK...) + API Key Secret
    """
    if not TWILIO_SID or not TWILIO_SID.startswith("AC"):
        raise ValueError(
            "TWILIO_ACCOUNT_SID must be your Account SID starting with 'AC' "
            f"(got {TWILIO_SID[:2] + '...' if TWILIO_SID else 'empty'}). "
            "Values starting with 'SK' are API Key SIDs — put those in "
            "TWILIO_API_KEY and keep TWILIO_ACCOUNT_SID as AC...."
        )

    api_key = TWILIO_API_KEY or ""
    api_secret = TWILIO_API_SECRET or ""
    if api_key.startswith("SK") and api_secret:
        # API Key auth: username=SK..., password=secret, account_sid=AC...
        return Client(api_key, api_secret, account_sid=TWILIO_SID)

    if not TWILIO_TOKEN:
        raise ValueError(
            "Set TWILIO_AUTH_TOKEN (Account Auth Token), or set both "
            "TWILIO_API_KEY (SK...) and TWILIO_API_SECRET."
        )
    return Client(TWILIO_SID, TWILIO_TOKEN)


def call_phone_with_twilio(room_name, prospect_token):
    """
    Uses Twilio to call MY_NUMBER.
    When answered, connects the call to the LiveKit room
    where Aiva is already waiting.
    """
    client = _twilio_client()

    # TwiML that connects the phone call to LiveKit
    # LiveKit has a native Twilio connector
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{LIVEKIT_TWILIO_WS_URL}">
            <Parameter name="token" value="{prospect_token}"/>
            <Parameter name="room" value="{room_name}"/>
        </Stream>
    </Connect>
</Response>"""

    call = client.calls.create(
        to    = MY_NUMBER,
        from_ = TWILIO_NUMBER,
        twiml = twiml,
    )

    print(f"Twilio call initiated: {call.sid}")
    return call


async def main():
    missing = [
        name for name, val in [
            ("LIVEKIT_URL", LIVEKIT_URL),
            ("LIVEKIT_API_KEY", LIVEKIT_KEY),
            ("LIVEKIT_API_SECRET", LIVEKIT_SECRET),
            ("TWILIO_ACCOUNT_SID", TWILIO_SID),
            ("TWILIO_AUTH_TOKEN", TWILIO_TOKEN),
            ("TWILIO_PHONE_NUMBER", TWILIO_NUMBER),
            ("MY_PHONE_NUMBER", MY_NUMBER),
        ] if not val
    ]
    if missing:
        print(f"ERROR: missing required .env values: {', '.join(missing)}")
        return

    room_name = f"porter-call-{int(time.time())}"

    prospect_token = await create_room_and_dispatch_aiva(room_name)
    call_phone_with_twilio(room_name, prospect_token)

    print("Make sure agent.py (the Aiva worker) is running so it can join the room.")


if __name__ == "__main__":
    asyncio.run(main())
