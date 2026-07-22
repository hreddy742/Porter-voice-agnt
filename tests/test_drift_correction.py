# ponytail: deterministic regression for deferred no-tool drift correction.

from types import SimpleNamespace

import agent_v2
from agent_v2 import Aiva
from livekit.agents.llm import ChatMessage


class FakeChatContext:
    def __init__(self, user_turn_count):
        self.items = [
            ChatMessage(role="user", content=[f"user turn {n}"])
            for n in range(user_turn_count)
        ]
        self.added = []

    def add_message(self, **kwargs):
        self.added.append(kwargs)


def _make_aiva(stage, user_turn_count=3):
    aiva = Aiva.__new__(Aiva)
    aiva.current_stage = stage
    aiva._call_ending = False
    aiva._user_turn_count = user_turn_count
    aiva._last_agent_speech_turn_count = user_turn_count
    aiva._suppress_stamp_update = False
    aiva._last_generation_had_tool_call = False
    aiva._no_tool_turn_streak = agent_v2.NO_TOOL_DRIFT_LIMIT - 1
    aiva._repeat_request_streak = 0
    aiva._last_notool_assistant_text = "A completely different prior response."
    aiva._pending_drift_correction = None
    return aiva


def _cross_drift_threshold(aiva):
    event = SimpleNamespace(
        item=ChatMessage(
            role="assistant",
            content=["Are you happy with your provider, or has it been a struggle?"],
        )
    )
    # This deliberately runs without an event loop. The old implementation
    # called asyncio.create_task() here and therefore created the extra reply.
    aiva._on_conversation_item_added(event)


def _check_stage(stage, expected_tool):
    aiva = _make_aiva(stage)
    _cross_drift_threshold(aiva)

    assert aiva._pending_drift_correction == (stage, 3)
    assert aiva._no_tool_turn_streak == 0

    # An explicit/system generation before another user turn must not consume
    # the correction or create any guidance.
    chat_ctx = FakeChatContext(user_turn_count=3)
    aiva._apply_pending_drift_correction(chat_ctx)
    assert chat_ctx.added == []
    assert aiva._pending_drift_correction == (stage, 3)

    # The next real user message makes the correction part of that one normal
    # response. It guides rather than forces, so unclear input cannot be
    # fabricated into a business outcome.
    chat_ctx.items.append(ChatMessage(role="user", content=["new answer"]))
    aiva._apply_pending_drift_correction(chat_ctx)
    assert aiva._pending_drift_correction is None
    assert len(chat_ctx.added) == 1
    correction = chat_ctx.added[0]
    assert correction["role"] == "system"
    assert expected_tool in correction["content"]
    assert "do not guess" in correction["content"].lower()

    # One pending correction is consumed once, never duplicated.
    aiva._apply_pending_drift_correction(chat_ctx)
    assert len(chat_ctx.added) == 1


def _check():
    _check_stage("qualifying", "qualifying_result")
    _check_stage("objection", "objection_result")

    aiva = _make_aiva("qualifying")
    aiva._pending_drift_correction = ("qualifying", 3)
    aiva.current_stage = "booking"
    chat_ctx = FakeChatContext(user_turn_count=4)
    aiva._apply_pending_drift_correction(chat_ctx)
    assert aiva._pending_drift_correction is None
    assert chat_ctx.added == []

    print(f"{agent_v2.__name__}: deferred drift-correction checks passed")


if __name__ == "__main__":
    _check()
