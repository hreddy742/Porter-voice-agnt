# TEMP investigation: does a totally clean opener call (no repeats, no
# wrong numbers) trip the NO-TOOL DRIFT GUARD purely from its own legitimate
# 3-turn shape (TURN 1 -> RIGHT-PERSON CHECK -> TURN 2)?
import asyncio
import agent_v2
from agent_v2 import Aiva
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
        for ev in result.events:
            item = getattr(ev, "item", None)
            if item is None: continue
            if item.type == "message" and item.role == "assistant":
                print(f"AIVA: {item.text_content}")
            elif item.type == "function_call":
                print(f"[tool call] {item.name}({item.arguments})")
            elif item.type == "function_call_output":
                print(f"[tool output] {item.output}")
        print(f"    -- streak={aiva._no_tool_turn_streak} user_turn_count={aiva._user_turn_count} call_ending={aiva._call_ending}")

    await turn("Hello?")                       # TURN 1 asked
    await turn("Yes it is.")                   # -> RIGHT-PERSON CHECK asked
    await turn("Yeah, that's me, I handle it.") # -> TURN 2 (disclosure+ask) should be asked here
    await asyncio.sleep(1.5)
    print(f"[FINAL] call_result={aiva.call_result!r} call_ending={aiva._call_ending} delete_room_called={ctx.delete_room_called}")
    await session.aclose()

asyncio.run(main())
