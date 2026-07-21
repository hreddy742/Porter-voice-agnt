# MANUAL, LIVE diagnostic for the forced tool_choice drift-correction fix
# (_inject_drift_correction / _STAGE_INSUFFICIENT_INFO_VALUE in agent_v2.py
# — see UPGRADE_NOTES.md). NOT part of the automated/deterministic test
# suite: makes real gpt-4o-mini API calls (needs OPENAI_API_KEY), costs
# money, and LLM output isn't perfectly deterministic, so a single run is
# not a pass/fail regression check — it's a rerunnable tool for
# re-verifying real model behavior (e.g. after a provider/prompt change, or
# investigating a new drift-correction incident in production).
#
# Uses the REAL Aiva agent, the real AgentSession/_on_conversation_item_added
# wiring — no mocks except hiding the qualifying_result tool for the setup
# turns (so the model genuinely cannot call it, guaranteeing the same "N
# no-tool turns" state a real drifted call reaches, without hoping the model
# avoids the tool on its own for the setup phase).
#
# Run: python test_drift_correction_live_manual.py

import asyncio

import agent_v2
from agent_v2 import Aiva, NO_TOOL_DRIFT_LIMIT
from livekit.agents.voice.agent_session import AgentSession


async def run_scenario(name, lead, user_turns):
    print(f"\n{'='*70}\nSCENARIO: {name}\n{'='*70}")

    aiva = Aiva(lead, ctx=None)
    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage("qualifying")

    # hide qualifying_result for the setup turns so the model genuinely
    # cannot call it (guarantees N no-tool turns instead of hoping the model
    # avoids a tool that's actually in scope). Production's own no-tool-drift
    # auto-trigger (asyncio.create_task(self._inject_drift_correction()),
    # wired via _on_conversation_item_added) still fires the instant the
    # streak crosses NO_TOOL_DRIFT_LIMIT — mid-loop, while the tool is still
    # hidden here — so it must be neutered for the setup phase or it forces
    # a tool_choice naming a tool absent from the registered set and the API
    # 400s. That's a test-harness artifact (current_stage and registered
    # tools never actually desync like this in production, since _set_stage
    # updates both atomically), not a production bug — noop it during setup,
    # then call the real method explicitly once for the actual check below.
    await aiva.update_tools([aiva.enter_disclosure, aiva.enter_exit])
    real_inject = aiva._inject_drift_correction
    async def _noop():
        pass
    aiva._inject_drift_correction = _noop

    assert len(user_turns) == NO_TOOL_DRIFT_LIMIT, "must match NO_TOOL_DRIFT_LIMIT exactly"

    for i, msg in enumerate(user_turns, 1):
        print(f"\n--- user turn {i} ---\nPROSPECT: {msg}")
        result = await session.run(user_input=msg)
        for ev in result.events:
            item = getattr(ev, "item", None)
            if item is None:
                continue
            if item.type == "message" and item.role == "assistant":
                print(f"AIVA: {item.text_content}")
            elif item.type == "function_call":
                print(f"[unexpected tool call during setup turn] {item.name}({item.arguments})")

    print(f"\n[state] no_tool_turn_streak={aiva._no_tool_turn_streak} "
          f"user_turn_count={aiva._user_turn_count} "
          f"last_agent_speech_turn_count={aiva._last_agent_speech_turn_count} "
          f"current_stage={aiva.current_stage}")

    # restore the real stage tools (qualifying_result back in scope) —
    # matches production: the tool was always registered, we only hid it to
    # force the setup turns to be tool-free. Leave aiva._inject_drift_correction
    # noop'd for the REST of this scenario's lifetime (don't restore it as the
    # instance attribute) — the background silence-escalation loop keeps the
    # session alive after our scripted turns end, which can trip the real
    # no-tool-drift streak again and race a second, redundant auto-triggered
    # correction against the one explicit call below. Calling the captured
    # `real_inject` bound method directly exercises the exact same real code,
    # once, deterministically, without that race.
    await aiva.update_tools(aiva._current_tools())

    # let any trailing background speech from the setup turns (rate-guardrail
    # retries, silence-escalation follow-ups) finish before capturing, so the
    # monkeypatch below only sees the forced call we're actually testing.
    await asyncio.sleep(1.5)

    print("\n--- forced drift correction fires ---")
    orig_generate_reply = session.generate_reply
    calls = []
    def _capture(**kwargs):
        handle = orig_generate_reply(**kwargs)
        calls.append({"tool_choice": kwargs.get("tool_choice"), "handle": handle})
        return handle
    session.generate_reply = _capture
    await real_inject()
    session.generate_reply = orig_generate_reply

    for c in calls:
        print(f"generate_reply call captured, tool_choice={c['tool_choice']}")

    print("\n--- full chat_ctx tail after the forced call (ground truth) ---")
    for item in aiva.chat_ctx.items[-8:]:
        if item.type == "message":
            print(f"  [{item.role}] {item.text_content}")
        elif item.type == "function_call":
            print(f"  [function_call] {item.name}({item.arguments})")
        elif item.type == "function_call_output":
            print(f"  [function_call_output] {item.output}")

    print(f"\n[state after forced call] current_stage={aiva.current_stage} "
          f"no_tool_turn_streak={aiva._no_tool_turn_streak} "
          f"violation_count={aiva._violation_count}")

    await session.aclose()


async def main():
    lead = {
        "company_name": "Summit Staffing Solutions",
        "city": "Tulsa",
        "state": "OK",
        "industry": "staffing",
        "tier": "warm",
    }

    # SCENARIO A: genuinely enough real info already given across 3 turns —
    # a model that just kept chatting instead of calling qualifying_result.
    await run_scenario(
        "A - info already present, model just avoided the tool",
        lead,
        [
            "Yeah so we're a staffing company, mostly light industrial and "
            "warehouse placements around the Tulsa area.",
            "We've actually been factoring with another company for about "
            "eight months now, they're charging us close to 2% and honestly "
            "the turnaround on funding has been kind of slow.",
            "We do maybe half a million a month in invoices, and yeah we'd "
            "definitely consider switching if the rate or the speed was better.",
        ],
    )

    # SCENARIO B: vague, insufficient info — forcing fires before the model
    # has anything real to base an outcome on.
    await run_scenario(
        "B - vague/insufficient info when forced fires",
        lead,
        [
            "Oh, um, yeah I guess so.",
            "I'm not really sure, it depends I think.",
            "Maybe, I'd have to check with my partner honestly.",
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
