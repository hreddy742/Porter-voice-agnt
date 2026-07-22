import asyncio
from types import SimpleNamespace

from agent_v2 import (
    Aiva,
    BOOKING_OPENING_LINE,
    PITCH_OPENING_LINE,
    QUALIFYING_OPENING_LINE,
)


class FakeSession:
    def __init__(self):
        self.said = []
        self.generated = []

    async def say(self, text, **kwargs):
        self.said.append((text, kwargs))

    async def generate_reply(self, **kwargs):
        self.generated.append(kwargs)


def _make_aiva(stage):
    aiva = Aiva.__new__(Aiva)
    aiva.current_stage = stage
    aiva._call_ending = False
    aiva._opener_result_armed_at_user_turn = None
    aiva._user_turn_count = 1
    aiva.call_result = "no_answer"
    aiva._activity = SimpleNamespace(session=FakeSession())
    transitions = []

    async def confirmed_turn():
        return True

    async def set_stage(new_stage):
        transitions.append(new_stage)
        aiva.current_stage = new_stage

    aiva._require_confirmed_turn = confirmed_turn
    aiva._set_stage = set_stage
    return aiva, transitions


async def _check_tool(tool_name, stage, args, expected_stage, expected_text):
    aiva, transitions = _make_aiva(stage)
    tool = getattr(Aiva, tool_name)
    result = await tool._func(aiva, *args)
    assert result is None
    assert transitions == [expected_stage]
    assert aiva.session.said == [
        (expected_text, {"allow_interruptions": True})
    ]
    assert aiva.session.generated == [], "deterministic transition used a second LLM"


async def _check():
    await _check_tool(
        "opener_result", "opener", ("interested",), "pitch", PITCH_OPENING_LINE
    )
    await _check_tool(
        "pitch_result", "pitch", ("interested",), "qualifying", QUALIFYING_OPENING_LINE
    )
    await _check_tool(
        "qualifying_result", "qualifying", ("qualified",), "booking", BOOKING_OPENING_LINE
    )
    await _check_tool(
        "objection_result", "objection", ("wants_callback",), "booking", BOOKING_OPENING_LINE
    )
    await _check_tool(
        "disclosure_result", "disclosure", ("wants_human",), "booking", BOOKING_OPENING_LINE
    )
    print("agent_v2: deterministic transition speech checks passed")


if __name__ == "__main__":
    asyncio.run(_check())
