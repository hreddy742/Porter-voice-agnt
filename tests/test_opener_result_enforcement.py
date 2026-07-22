from livekit.agents.llm import ChatMessage

from agent_v2 import (
    Aiva,
    OPENER_DISCLOSURE_CLARIFICATION_LINE,
    OPENER_DISCLOSURE_LINE,
    _is_opener_disclosure,
)


class FakeChatContext:
    def __init__(self, user_turn_count):
        self.items = [
            ChatMessage(role="user", content=[f"user turn {index}"])
            for index in range(user_turn_count)
        ]


def _make_aiva(stage="opener", armed_at=2):
    aiva = Aiva.__new__(Aiva)
    aiva.current_stage = stage
    aiva._opener_result_armed_at_user_turn = armed_at
    return aiva


def _check():
    assert _is_opener_disclosure(
        "I'll be upfront — this is a cold call, and I'm an AI calling for Porter Capital."
    )
    assert not _is_opener_disclosure("We can connect you with an AI advisor.")
    assert "cold call" in OPENER_DISCLOSURE_LINE
    assert "AI assistant" in OPENER_DISCLOSURE_LINE
    assert OPENER_DISCLOSURE_CLARIFICATION_LINE.endswith("30 seconds?")

    aiva = _make_aiva()
    assert not aiva._opener_result_is_due(FakeChatContext(2))
    assert aiva._opener_result_is_due(FakeChatContext(3))

    # The exact production response after disclosure must require resolution.
    assert aiva._opener_result_is_due(FakeChatContext(3)), "ten-second reply must resolve opener"

    # A stale arm cannot affect another stage, and an unarmed opener is inert.
    assert not _make_aiva("pitch")._opener_result_is_due(FakeChatContext(3))
    assert not _make_aiva(armed_at=None)._opener_result_is_due(FakeChatContext(3))

    print("agent_v2: opener-result enforcement checks passed")


if __name__ == "__main__":
    _check()
