# TEMP investigation: does a totally clean opener call (no repeats, no
# wrong numbers) trip the NO-TOOL DRIFT GUARD purely from its own legitimate
# 3-turn shape (TURN 1 -> RIGHT-PERSON CHECK -> TURN 2)?
import asyncio
import agent_v2
from agent_v2 import (
    Aiva,
    OPENER_DISCLOSURE_CLARIFICATION_LINE,
    OPENER_DISCLOSURE_LINE,
    PITCH_OPENING_LINE,
)
from livekit.agents.voice.agent_session import AgentSession

class _FakeCtx:
    def __init__(self): self.delete_room_called = False
    async def delete_room(self): self.delete_room_called = True

async def main():
    lead = {"company_name": "Riverbend Logistics", "city": "Amarillo", "state": "TX",
            "contact_name": "Sam", "industry": "trucking", "tier": "warm"}
    ctx = _FakeCtx()
    aiva = Aiva(lead, ctx=ctx)
    async def _noop(): pass
    aiva.on_enter = _noop

    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage("opener")

    async def turn(user_input):
        result = await session.run(user_input=user_input)
        assistant_messages = []
        for ev in result.events:
            item = getattr(ev, "item", None)
            if item is None: continue
            if item.type == "message" and item.role == "assistant":
                assistant_messages.append(item.text_content)
                print(f"AIVA: {item.text_content}")
            elif item.type == "function_call":
                print(f"[tool call] {item.name}({item.arguments})")
            elif item.type == "function_call_output":
                print(f"[tool output] {item.output}")
        print(f"    -- streak={aiva._no_tool_turn_streak} user_turn_count={aiva._user_turn_count} call_ending={aiva._call_ending}")
        return assistant_messages

    await turn("Hello?")                       # TURN 1 asked
    await turn("Yes it is.")                   # -> RIGHT-PERSON CHECK asked
    disclosure_messages = await turn("Yeah, that's me, I handle it.")
    assert disclosure_messages == [OPENER_DISCLOSURE_LINE]
    assert aiva._opener_result_armed_at_user_turn is not None
    pitch_messages = await turn("I just got ten seconds for you.")
    assert pitch_messages == [PITCH_OPENING_LINE]
    assert aiva.current_stage == "pitch"
    assert aiva._opener_result_armed_at_user_turn is None
    await asyncio.sleep(1.5)
    print(f"[FINAL] call_result={aiva.call_result!r} call_ending={aiva._call_ending} delete_room_called={ctx.delete_room_called}")
    await session.aclose()

    # A clarification after disclosure must remain in opener and rephrase;
    # forcing the tool must never guess interest or rejection.
    clarify_ctx = _FakeCtx()
    clarify_aiva = Aiva(lead, ctx=clarify_ctx)
    clarify_aiva.on_enter = _noop
    clarify_session = AgentSession(llm=agent_v2._build_llm("gpt"))
    clarify_session.on(
        "conversation_item_added", clarify_aiva._on_conversation_item_added
    )
    await clarify_session.start(agent=clarify_aiva)
    await clarify_aiva._set_stage("opener")
    clarify_aiva._opener_result_armed_at_user_turn = 0
    result = await clarify_session.run(user_input="Sorry, what do you mean?")
    tool_calls = [
        event.item
        for event in result.events
        if getattr(event, "item", None) is not None
        and event.item.type == "function_call"
    ]
    clarification_messages = [
        event.item.text_content
        for event in result.events
        if getattr(event, "item", None) is not None
        and event.item.type == "message"
        and event.item.role == "assistant"
        and event.item.text_content
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "opener_result"
    assert '"needs_clarification"' in tool_calls[0].arguments
    assert clarification_messages == [OPENER_DISCLOSURE_CLARIFICATION_LINE]
    assert clarify_aiva.current_stage == "opener"
    assert not clarify_aiva._call_ending
    await clarify_session.aclose()

    reject_ctx = _FakeCtx()
    reject_aiva = Aiva(lead, ctx=reject_ctx)
    reject_aiva.on_enter = _noop
    reject_session = AgentSession(llm=agent_v2._build_llm("gpt"))
    reject_session.on(
        "conversation_item_added", reject_aiva._on_conversation_item_added
    )
    await reject_session.start(agent=reject_aiva)
    await reject_aiva._set_stage("opener")
    reject_aiva._opener_result_armed_at_user_turn = 0
    result = await reject_session.run(user_input="No thanks, I'm not interested.")
    tool_calls = [
        event.item
        for event in result.events
        if getattr(event, "item", None) is not None
        and event.item.type == "function_call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "opener_result"
    assert '"not_interested"' in tool_calls[0].arguments
    assert reject_aiva.call_result == "not_interested"
    assert reject_aiva._call_ending
    await reject_session.aclose()

asyncio.run(main())
