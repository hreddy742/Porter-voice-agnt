"""Manual real-model check for stage-independent callback routing.

Uses the exact request from production call 27. Requires OPENAI_API_KEY and
costs money, so it is intentionally separate from deterministic tests.
"""

import asyncio

import agent_v2
from agent_v2 import Aiva
from livekit.agents.voice.agent_session import AgentSession


class FakeJobContext:
    async def delete_room(self):
        pass


async def main():
    aiva = Aiva(
        {
            "company_name": "AB Staffing Solutions LLC",
            "city": "",
            "state": "",
            "industry": "staffing",
            "tier": "warm",
        },
        ctx=FakeJobContext(),
    )

    async def skip_hello():
        pass

    aiva.on_enter = skip_hello
    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage("opener")

    result = await session.run(
        user_input=(
            "Could you set up a call with one of the sales reps or the adviser?"
        )
    )
    assistant_messages = [
        event.item.text_content
        for event in result.events
        if getattr(event, "item", None) is not None
        and event.item.type == "message"
        and event.item.role == "assistant"
        and event.item.text_content
    ]

    assert aiva.current_stage == "booking"
    assert not aiva._call_ending
    assert len(assistant_messages) <= 1
    assert not any("right person" in text.lower() for text in assistant_messages)
    assert not any("have a great day" in text.lower() for text in assistant_messages)
    print("assistant:", assistant_messages)
    print("live callback-intent routing check passed")
    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main())
