"""LiveKit worker entrypoint — session lifecycle and CLI runner."""

import asyncio
import json
import os
import time
import urllib.request

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    JobContext,
    JobExecutorType,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, silero

from app.agent.providers import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, get_llm, get_tts
from app.agent.supervisor import PorterSupervisor
from app.db import get_next_lead, update_lead_status

load_dotenv()

current_lead = None


def prewarm_qwen(proc: JobProcess) -> None:
    """Load qwen3:4b into Ollama's memory before any job needs it.

    qwen is only ever reached via FallbackAdapter when Groq fails mid-call
    (see get_llm()), and a cold load measured ~24s+ — dead air on a live
    call. WorkerOptions.prewarm_fnc runs once per worker process, before
    that process is handed a job, so this pays the cold-load cost at
    startup instead of mid-conversation. No-ops if qwen isn't the
    configured failover.
    """
    if os.getenv("LLM_FAILOVER_PROVIDER", "").lower() != "qwen":
        return

    ollama_root = OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
    start = time.monotonic()
    try:
        req = urllib.request.Request(
            f"{ollama_root}/api/generate",
            data=json.dumps({
                "model": "qwen3:4b",
                "prompt": "hi",
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"num_predict": 1},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=60)
        print(f"qwen3:4b warm-up complete in {time.monotonic() - start:.1f}s (keep_alive={OLLAMA_KEEP_ALIVE})", flush=True)
    except Exception as e:
        print(f"WARNING: qwen3:4b warm-up failed ({e}); first real failover call may be cold", flush=True)


async def when_call_starts(ctx: JobContext):

    global current_lead

    lead_model = get_next_lead()

    if lead_model is None:
        print("No leads available.")
        return

    # Tasks / supervisor expect dict-style `.get()` access.
    lead = lead_model.model_dump()
    current_lead = lead
    print(f"Calling: {lead['company_name']} | Tier: {lead['tier']}")

    pipeline = AgentSession(
        stt = deepgram.STT(model="nova-2"),
        llm = get_llm(),
        tts = get_tts(),
        vad = silero.VAD.load(),
    )

    supervisor = PorterSupervisor(lead, ctx)

    # generate_reply() only dispatches the current turn — it returns long
    # before the AgentTask chain (opener -> qualifier/objection/booking/exit)
    # actually reaches a terminal call_result. The session only truly ends
    # when AgentSession emits "close" (task completed, participant hung up,
    # error, etc. — see livekit.agents.voice.events.CloseReason), so wait
    # on that event before reading call_result / touching the DB.
    call_ended = asyncio.Event()
    pipeline.on("close", lambda ev: call_ended.set())

    # Aiva must never go dead silent. "error" fires for every failed LLM
    # generation attempt, but error.recoverable=False only once the whole
    # attempt is exhausted (all retries AND the FallbackAdapter's full
    # Groq -> qwen chain) — i.e. exactly the turns that would otherwise
    # produce no spoken reply at all. Speak a recovery line directly via
    # say(), bypassing the LLM, so the prospect always hears something.
    def _on_agent_error(ev):
        error = ev.error
        if getattr(error, "type", None) == "llm_error" and not getattr(error, "recoverable", True):
            print(f"LLM generation failed after all retries/fallback: {error.error}")
            pipeline.say(
                "Sorry, could you say that again?",
                allow_interruptions=True,
            )

    pipeline.on("error", _on_agent_error)

    await pipeline.start(
        room               = ctx.room,
        agent              = supervisor,
        room_input_options = RoomInputOptions(),
    )

    # Must go through generate_reply()/tool-calling, NOT a direct
    # `await supervisor.start_call()` — AgentTask.__await_impl() (see
    # livekit.agents.voice.agent.py) requires the awaiting asyncio.Task to be
    # tagged inline_task=True, and that tag is only ever set on two kinds of
    # task: the SDK's own tool-function dispatch task, and the on_enter/
    # on_exit task created by AgentActivity.start(). when_call_starts's task
    # is neither, so a direct call makes OpenerTask's internal
    # `await OpenerTask(...)` raise RuntimeError. Going through
    # generate_reply() lets the LLM invoke start_call as a real tool call,
    # which runs inside the SDK's inline-tagged dispatch task. The
    # wait-for-their-hello-or-timeout race still happens correctly once
    # inside OpenerTask.on_enter() — that context was never the problem.
    await pipeline.generate_reply(
        instructions="Call just connected. Call start_call now. Nothing else."
    )

    await call_ended.wait()

    if os.getenv("TEST_MODE", "false").lower() == "true":
        print(f"TEST MODE — lead status NOT updated. Would have set: {supervisor.call_result}")
    else:
        update_lead_status(
            lead_candidate_id = lead['lead_candidate_id'],
            new_status        = supervisor.call_result
        )

    print(f"Call ended. Result: {supervisor.call_result}")


def run_worker() -> None:
    """Start the LiveKit Agents worker (console / dev / start)."""
    cli.run_app(WorkerOptions(
        entrypoint_fnc=when_call_starts,
        prewarm_fnc=prewarm_qwen,
        job_executor_type=JobExecutorType.PROCESS,
        # dev mode defaults to 0 idle processes, which means prewarm_fnc
        # wouldn't run until the first job arrives — too late to matter.
        # Force at least one warm process so qwen loads at worker startup.
        num_idle_processes=1,
        # Default initialize_process_timeout (10s) is shorter than
        # prewarm_qwen's observed 24s+ cold-load of qwen3:4b, so the SDK
        # was killing and immediately respawning the idle process forever
        # instead of ever reaching a warm, ready state. 40s gives the same
        # kind of margin above observed time as OPENER_WAIT_TIMEOUT_SECONDS
        # does above typical response latency.
        initialize_process_timeout=40.0,
    ))


if __name__ == "__main__":
    run_worker()

