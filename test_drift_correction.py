# ponytail: minimal self-check for the forced tool_choice drift-correction
# fix (_inject_drift_correction in agent_v2.py). Verifies the plumbing this
# change touches: the right *_result tool is force-selected per stage, and
# the guard conditions (call ending / no decision point for this stage)
# still skip correctly. Does NOT make a live LLM call — whether OpenAI/Groq
# actually honor named tool_choice is confirmed via their docs and the
# installed livekit-plugins-groq source (a thin subclass of the OpenAI
# plugin), not re-verified here. Run: python test_drift_correction.py

import asyncio

import agent_v2
from agent_v2 import Aiva, _STAGE_DECISION_POINTS, _STAGE_INSUFFICIENT_INFO_VALUE

# Aiva.session is a read-only property (inherited from Agent) that reaches
# into a live LiveKit activity/session we don't have in a unit test. Shadow
# it on Aiva with a plain settable property, backed by _fake_session, for
# this process only.
Aiva.session = property(
    lambda self: self._fake_session,
    lambda self, v: setattr(self, "_fake_session", v),
)


class FakeSession:
    def __init__(self):
        self.calls = []

    async def generate_reply(self, **kwargs):
        self.calls.append(kwargs)


def _make_aiva(user_turn_count=5):
    aiva = Aiva.__new__(Aiva)  # skip __init__ (needs a real JobContext/session)
    aiva._call_ending = False
    aiva.session = FakeSession()
    aiva._user_turn_count = user_turn_count
    aiva._last_agent_speech_turn_count = user_turn_count  # as if just re-stamped
    return aiva


def _check():
    # every real stage forces exactly its own *_result tool, and rolls the
    # confirmed-turn stamp back one user turn (see _inject_drift_correction)
    for stage, (tool_name, _decision) in _STAGE_DECISION_POINTS.items():
        aiva = _make_aiva()
        aiva.current_stage = stage
        asyncio.run(aiva._inject_drift_correction())
        assert len(aiva.session.calls) == 1, f"{stage}: expected one generate_reply call"
        tool_choice = aiva.session.calls[0].get("tool_choice")
        assert tool_choice == {"type": "function", "function": {"name": tool_name}}, (
            f"{stage}: tool_choice was {tool_choice!r}, expected forced {tool_name!r}"
        )
        assert aiva._last_agent_speech_turn_count == aiva._user_turn_count - 1, (
            f"{stage}: stamp was not rolled back to just before the last real user turn"
        )

        instructions = aiva.session.calls[0].get("instructions", "")
        insufficient_value = _STAGE_INSUFFICIENT_INFO_VALUE.get(stage)
        if insufficient_value:
            assert insufficient_value in instructions, (
                f"{stage}: expected the '{insufficient_value}' escape hatch "
                "mentioned in the corrective instructions"
            )
        else:
            assert "rather than guessing" not in instructions, (
                f"{stage}: has no insufficient-info value yet, must not "
                "reference an escape hatch that doesn't exist"
            )

    # a stage with no decision point (e.g. "hello") must not force anything
    aiva = _make_aiva()
    aiva.current_stage = "hello"
    asyncio.run(aiva._inject_drift_correction())
    assert aiva.session.calls == [], "hello has no *_result tool, must not force one"

    # _call_ending must short-circuit before any generate_reply call
    aiva = _make_aiva()
    aiva.current_stage = "pitch"
    aiva._call_ending = True
    asyncio.run(aiva._inject_drift_correction())
    assert aiva.session.calls == [], "_call_ending must skip the forced call entirely"

    print(f"{agent_v2.__name__}: all drift-correction tool_choice checks passed")


if __name__ == "__main__":
    _check()
