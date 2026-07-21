# MANUAL, LIVE stress test for the Exit-stage flow (enter_exit / _do_opt_out /
# _exit_block in agent_v2.py). Previously this relied on the model calling a
# SECOND tool (opt_out) in the same automatic continuation as enter_exit's
# acknowledgment — a model-judgment guarantee, not a code guarantee, and it
# turned out to be structurally broken: generate_reply() called from inside
# a function_tool defaults tool_choice to "none" for that generation (see
# livekit-agents' agent_activity.py, _generate_reply()), so opt_out was never
# actually reachable in that continuation. That's what the original 0/10 run
# caught. The fix: enter_exit() now speaks the acknowledgment, waits for
# playout, then calls the suppression/end-call logic directly in code
# (_do_opt_out()) — no second tool call, no model judgment involved. Not
# part of the automated/deterministic suite: makes real gpt-4o-mini API
# calls (needs OPENAI_API_KEY), costs money, LLM output isn't deterministic.
#
# SUCCESS CRITERION (stated before running, per task instructions):
# 10 trials, varied opt-out phrasing. A trial is CLEAN only if ALL of:
#   (a) the acknowledgment is spoken exactly once (not 0, not 2+)
#   (b) the call actually ends cleanly (call_result == "suppressed",
#       ctx.delete_room() invoked) — driven by code, not a second tool call
#   (c) no same-breath violation fires anywhere in this stage
# Given this drives a real compliance action (do-not-call suppression), the
# bar is 10/10 clean, not a majority — a single failure means "not proven
# yet" and gets reported, not smoothed over.
#
# Run: python test_exit_stage_stress_live_manual.py

import asyncio

import agent_v2
from agent_v2 import Aiva
from livekit.agents.voice.agent_session import AgentSession


class _FakeCtx:
    """Stub for JobContext — only delete_room() is touched by _end_call()."""
    def __init__(self):
        self.delete_room_called = False

    async def delete_room(self):
        self.delete_room_called = True


OPT_OUT_PHRASINGS = [
    "Please remove me from your call list.",
    "Take me off your list, I don't want any more calls.",
    "Can you stop calling this number, please.",
    "I'd like to unsubscribe / opt out from future calls.",
    "Don't call us again, take us off whatever list this is.",
    "Yeah please just remove our company from your calling list.",
    "Stop contacting us, we're not interested in ever hearing from you again.",
    "Can you guys take me off your calling list, please.",
    "I need you to remove this number from your outreach list.",
    "Please don't call this number anymore, take it off your list.",
]


async def run_trial(i, phrase):
    print(f"\n{'='*70}\nTRIAL {i}: {phrase!r}\n{'='*70}")

    lead = {
        "company_name": "Riverbend Logistics",
        "city": "Amarillo",
        "state": "TX",
        "industry": "trucking",
        "tier": "warm",
    }
    ctx = _FakeCtx()
    aiva = Aiva(lead, ctx=ctx)

    # on_enter() (the hello-escalation loop) auto-starts as a background
    # task the instant session.start() runs — it has no idea this test is
    # about to skip straight past "hello" into "qualifying". Left alone it
    # keeps firing its own generate_reply() calls on a 3.5s timer, racing
    # against our scripted turn and contaminating the transcript with
    # spurious tool calls unrelated to the Exit fix under test. Same
    # test-harness-artifact class as test_drift_correction_live_manual.py's
    # neutered _inject_drift_correction — not a production bug, since
    # production always genuinely starts at "hello".
    async def _noop_on_enter():
        pass
    aiva.on_enter = _noop_on_enter

    session = AgentSession(llm=agent_v2._build_llm("gpt"))
    session.on("conversation_item_added", aiva._on_conversation_item_added)
    await session.start(agent=aiva)
    await aiva._set_stage("qualifying")

    same_breath_events = []
    real_require_confirmed_turn = aiva._require_confirmed_turn

    async def _wrapped_require_confirmed_turn():
        same_breath_events.append(aiva._same_breath_violation)
        return await real_require_confirmed_turn()

    aiva._require_confirmed_turn = _wrapped_require_confirmed_turn

    result = await session.run(user_input=phrase)
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

    # give any trailing generate_reply from _end_call/opt_out a moment to land
    await asyncio.sleep(1.0)

    # ---- verdict, from ground-truth chat_ctx, not just this run()'s events ----
    # opt_out no longer exists as a separate tool call — _do_opt_out() runs
    # as plain code inside enter_exit() right after the acknowledgment plays,
    # so there's nothing to look up by name for it. The acknowledgment count
    # and call-ending are the observable proxies for "the whole flow ran".
    items = aiva.chat_ctx.items
    enter_exit_idx = next(
        (n for n, it in enumerate(items) if it.type == "function_call" and it.name == "enter_exit"),
        None,
    )

    ack_count = 0
    if enter_exit_idx is not None:
        ack_count = sum(
            1 for it in items[enter_exit_idx + 1:]
            if it.type == "message" and it.role == "assistant" and it.text_content and it.text_content.strip()
        )

    called_enter_exit = enter_exit_idx is not None
    same_breath_fired = any(same_breath_events)
    call_ended_clean = ctx.delete_room_called and aiva.call_result == "suppressed"

    clean = (
        called_enter_exit
        and ack_count == 1
        and not same_breath_fired
        and call_ended_clean
    )

    print(
        f"\n[verdict] enter_exit_called={called_enter_exit} "
        f"ack_count={ack_count} same_breath_fired={same_breath_fired} "
        f"call_result={aiva.call_result!r} delete_room_called={ctx.delete_room_called} "
        f"=> {'CLEAN' if clean else 'FAIL'}"
    )

    await session.aclose()
    return {
        "phrase": phrase,
        "enter_exit_called": called_enter_exit,
        "ack_count": ack_count,
        "same_breath_fired": same_breath_fired,
        "call_result": aiva.call_result,
        "delete_room_called": ctx.delete_room_called,
        "clean": clean,
    }


async def main():
    results = []
    for i, phrase in enumerate(OPT_OUT_PHRASINGS, 1):
        results.append(await run_trial(i, phrase))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    clean_count = sum(1 for r in results if r["clean"])
    for i, r in enumerate(results, 1):
        status = "CLEAN" if r["clean"] else "FAIL"
        print(f"{i:2}. [{status}] ack_count={r['ack_count']} "
              f"delete_room_called={r['delete_room_called']} "
              f"same_breath_fired={r['same_breath_fired']} "
              f"call_result={r['call_result']!r} :: {r['phrase']!r}")
    print(f"\n{clean_count}/{len(results)} clean "
          f"({'PASS — Exit fix proven' if clean_count == len(results) else 'FAIL — Exit fix NOT proven, see failures above'})")


if __name__ == "__main__":
    asyncio.run(main())
