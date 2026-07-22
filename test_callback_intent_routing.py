import asyncio

from livekit.agents.llm import ChatMessage

from agent_v2 import Aiva


class FakeChatContext:
    def __init__(self, user_text):
        self.items = [ChatMessage(role="user", content=[user_text])]
        self.added = []

    def add_message(self, **message):
        self.added.append(message)


def _make_aiva(stage="opener"):
    aiva = Aiva.__new__(Aiva)
    aiva.current_stage = stage
    aiva._call_ending = False
    transitions = []

    async def set_stage(new_stage):
        transitions.append(new_stage)
        aiva.current_stage = new_stage

    aiva._set_stage = set_stage
    return aiva, transitions


def _check():
    # Exact intent from production call 27: stale opener must route to booking.
    aiva, transitions = _make_aiva()
    chat_ctx = FakeChatContext(
        "Could you set up a call with one of the sales reps or the adviser?"
    )
    assert asyncio.run(aiva._route_explicit_callback_request(chat_ctx)) is True
    assert transitions == ["booking"]
    assert "not a gatekeeper referral" in chat_ctx.added[0]["content"]

    # Common equivalent phrasing follows the same route.
    aiva, transitions = _make_aiva("qualifying")
    chat_ctx = FakeChatContext("I'd rather speak to a human.")
    assert asyncio.run(aiva._route_explicit_callback_request(chat_ctx)) is True
    assert transitions == ["booking"]

    # Mentioning an advisor without asking for a callback is not enough.
    aiva, transitions = _make_aiva()
    chat_ctx = FakeChatContext("What does your advisor usually review?")
    assert asyncio.run(aiva._route_explicit_callback_request(chat_ctx)) is False
    assert transitions == []
    assert chat_ctx.added == []

    # Booking is idempotent and terminal stages cannot be reopened.
    for stage in ("booking", "exit"):
        aiva, transitions = _make_aiva(stage)
        chat_ctx = FakeChatContext("Please schedule a call.")
        assert asyncio.run(aiva._route_explicit_callback_request(chat_ctx)) is False
        assert transitions == []

    print("agent_v2: explicit callback-intent routing checks passed")


if __name__ == "__main__":
    _check()
