# MANUAL, LIVE diagnostic for deferred no-tool drift correction.
# Uses the real GPT/LiveKit pipeline and costs money. It verifies that reaching
# the drift threshold never creates a second assistant turn, then supplies one
# more user answer and confirms the pending guidance is consumed by that one
# natural response. Run: python test_drift_correction_live_manual.py

import asyncio

import agent_v2
from agent_v2 import Aiva, NO_TOOL_DRIFT_LIMIT
from livekit.agents.voice.agent_session import AgentSession


class _FakeCtx:
    async def delete_room(self):
        pass


def _assistant_count(aiva):
    return sum(
        bool(item.type == "message" and item.role == "assistant" and item.text_content)
        for item in aiva.chat_ctx.items
    )


async def run_scenario(stage, setup_turns, final_turn):
    lead = {
        "company_name": "Summit Staffing Solutions",
        "city": "Tulsa",
        "state": "OK",
        "industry": "staffing",
        "tier": "warm",
    }
    aiva = Aiva(lead, ctx=_FakeCtx())

    async def _noop_on_enter():
        pass

    aiva.on_enter = _noop_on_enter
    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage(stage)

    stage_tool = getattr(aiva, agent_v2._STAGE_DECISION_POINTS[stage][0])
    await aiva.update_tools([aiva.enter_disclosure, aiva.enter_exit])
    assert len(setup_turns) == NO_TOOL_DRIFT_LIMIT

    for user_input in setup_turns:
        await session.run(user_input=user_input)

    count_at_threshold = _assistant_count(aiva)
    await asyncio.sleep(1.0)
    count_after_wait = _assistant_count(aiva)
    assert count_after_wait == count_at_threshold, (
        f"{stage}: drift threshold created an unsolicited assistant reply"
    )
    assert aiva._pending_drift_correction == (stage, len(setup_turns))

    await aiva.update_tools([stage_tool, aiva.enter_disclosure, aiva.enter_exit])
    before_final = _assistant_count(aiva)
    result = await session.run(user_input=final_turn)
    await asyncio.sleep(0.5)
    after_final = _assistant_count(aiva)

    for event in result.events:
        item = getattr(event, "item", None)
        if item is not None and item.type in {
            "message",
            "function_call",
            "function_call_output",
        }:
            print(stage, item.type, getattr(item, "text_content", None) or getattr(item, "name", ""))

    assert aiva._pending_drift_correction is None
    assert after_final - before_final <= 1, (
        f"{stage}: one user turn produced {after_final - before_final} assistant replies"
    )
    await session.aclose()


async def main():
    await run_scenario(
        "qualifying",
        [
            "We're an IT consulting business.",
            "We currently factor our invoices.",
            "We're happy with the provider, but I'm open to hearing your offer.",
        ],
        "Yes, go ahead and tell me what would be different.",
    )
    await run_scenario(
        "objection",
        [
            "We already use another factoring company.",
            "They've generally been fine for us.",
            "I can still hear what makes you different.",
        ],
        "Yes, I'm still interested in comparing the options.",
    )
    print("live deferred drift-correction checks passed")


if __name__ == "__main__":
    asyncio.run(main())
