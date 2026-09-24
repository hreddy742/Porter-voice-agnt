# ============================================================
# dialer.py — Make Aiva call a real lead
# Uses Twilio Programmable Voice (works on trial accounts)
# Usage: python call.py  (thin root wrapper)
# ============================================================

import os
import time
import asyncio
from dotenv import load_dotenv
from livekit import api
from twilio.rest import Client

from app.db import get_next_lead

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

# TEST_MODE dials MY_PHONE_NUMBER (your own phone) instead of the selected
# lead's real number. Either way, the room is tagged with the SAME
# lead_candidate_id (see create_room_and_dispatch_aiva), so a test call
# still exercises the real lead-selection and agent-context path end to
# end — only the number actually rung changes. Same TEST_MODE flag
# worker.py already uses to skip writing call_result back to the DB.
TEST_MODE      = os.getenv("TEST_MODE", "false").lower() == "true"
MY_NUMBER      = os.getenv("MY_PHONE_NUMBER")

# LiveKit Cloud's Twilio Media Streams connector for your project
LIVEKIT_TWILIO_WS_URL = os.getenv(
    "LIVEKIT_TWILIO_WS_URL",
    "wss://0fsr7yjvwys.sip.livekit.cloud/twilio",
)


async def create_room_and_dispatch_aiva(room_name, lead_candidate_id):
    """
    Creates a LiveKit room, tagged with the lead_candidate_id that was
    actually dialed. Aiva's worker process (python agent.py) is running
    with automatic dispatch, so it joins this room on its own as soon as
    the room exists — no explicit dispatch call is needed. Aiva will wait
    in the room until the phone call connects.

    The metadata tag is what lets worker.py's when_call_starts() pick up
    the SAME lead that Twilio actually dialed below, instead of
    independently querying get_next_lead() and risking a mismatch between
    who was called and who the agent thinks it's talking to.
    """
    lk = api.LiveKitAPI(
        url        = LIVEKIT_URL.replace("wss://", "https://"),
        api_key    = LIVEKIT_KEY,
        api_secret = LIVEKIT_SECRET,
    )

    # Create the room, tagged with the dialed lead's id
    await lk.room.create_room(
        api.CreateRoomRequest(name=room_name, metadata=lead_candidate_id)
    )
    print(f"Room created: {room_name} (lead_candidate_id={lead_candidate_id})")

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


def call_phone_with_twilio(room_name, prospect_token, to_number):
    """
    Uses Twilio to call `to_number`.
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
        to    = to_number,
        from_ = TWILIO_NUMBER,
        twiml = twiml,
    )

    print(f"Twilio call initiated: {call.sid} -> {to_number}")
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
        ] if not val
    ]
    if TEST_MODE and not MY_NUMBER:
        missing.append("MY_PHONE_NUMBER")
    if missing:
        print(f"ERROR: missing required .env values: {', '.join(missing)}")
        return

    # Select the lead BEFORE dialing, so the number Twilio rings and the
    # lead the agent has context on are guaranteed to be the same one —
    # see create_room_and_dispatch_aiva()'s metadata tag and
    # worker.py's when_call_starts(). This replaces the old behavior
    # where dialer.py always rang a fixed MY_PHONE_NUMBER regardless of
    # which lead (if any) the worker later picked for itself.
    lead = get_next_lead()
    if lead is None:
        print("No leads available to call.")
        return

    to_number = MY_NUMBER if TEST_MODE else lead.phone
    if not to_number:
        print(f"ERROR: lead {lead.lead_candidate_id} ({lead.company_name}) has no phone number.")
        return

    mode_note = f"TEST MODE -> {to_number}" if TEST_MODE else to_number
    print(f"Calling: {lead.company_name} | Tier: {lead.tier} | {mode_note}")

    room_name = f"porter-call-{int(time.time())}"

    prospect_token = await create_room_and_dispatch_aiva(room_name, lead.lead_candidate_id)
    call_phone_with_twilio(room_name, prospect_token, to_number)

    print("Make sure agent.py (the Aiva worker) is running so it can join the room.")


if __name__ == "__main__":
    asyncio.run(main())
