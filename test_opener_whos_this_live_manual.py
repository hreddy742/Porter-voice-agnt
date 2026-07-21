# MANUAL, LIVE regression test for the opener stage's "who's this?"
# handling (_opener_block()'s CALLER-IDENTITY QUESTION BEFORE TURN 2 rule in
# agent_v2.py). NOT part of the automated/deterministic suite: makes real
# gpt-4o-mini API calls (needs OPENAI_API_KEY), costs money, LLM output
# isn't perfectly deterministic.
#
# Live-reported bug (2026-07-21): a prospect confirming TURN 1 and asking
# "who's this?" in the same breath was completely ignored — the model moved
# straight into the RIGHT-PERSON CHECK as if the question was never asked.
# The opener instructions had no branch for this case at all (distinct from
# CONFUSED, which is about not having caught the company/contact name in
# TURN 1 itself). Fix: a dedicated instruction answers honestly ("I'm
# calling from Porter Capital") without the AI/cold-call disclosure, which
# stays reserved for TURN 2, then continues to the RIGHT-PERSON CHECK.
#
# SUCCESS CRITERION: the model acknowledges the identity question (names
# Porter Capital) and does NOT say "AI"/"cold call" before continuing to the
# RIGHT-PERSON CHECK in the same or next reply.
#
# Run: python test_opener_whos_this_live_manual.py

import asyncio

import agent_v2
from agent_v2 import Aiva
from livekit.agents.voice.agent_session import AgentSession


class _FakeCtx:
    def __init__(self):
        self.delete_room_called = False

    async def delete_room(self):
        self.delete_room_called = True


async def run_turn(session, aiva, user_input):
    result = await session.run(user_input=user_input)
    for ev in result.events:
        item = getattr(ev, "item", None)
        if item is None:
            continue
        if item.type == "message" and item.role == "assistant":
            print(f"AIVA: {item.text_content}")
        elif item.type == "function_call":
            print(f"[tool call] {item.name}({item.arguments})")
        elif item.type == "function_call_output":
            print(f"[tool output] {item.output}")


async def main():
    lead = {
        "company_name": "Riverbend Logistics",
        "city": "Amarillo",
        "state": "TX",
        "contact_name": "Sam",
        "industry": "trucking",
        "tier": "warm",
    }
    ctx = _FakeCtx()
    aiva = Aiva(lead, ctx=ctx)

    async def _noop_on_enter():
        pass
    aiva.on_enter = _noop_on_enter

    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage("opener")

    await run_turn(session, aiva, "Hello?")
    await run_turn(session, aiva, "Yes it is -- who's this?")

    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main())
