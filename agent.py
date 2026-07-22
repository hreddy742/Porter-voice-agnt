"""LiveKit worker entrypoint for the Porter Capital voice agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import cartesia, deepgram, openai

from app.agent.voice_agent import Aiva
from app.agent.prompt import build_context_variables
from app.config import Settings
from app.infrastructure.database import PostgresRepository


load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("porter.agent")
INACTIVITY_REMINDERS = 3
INACTIVITY_REMINDER_INTERVAL = 10.0
INACTIVITY_INSTRUCTION = (
    "The user has been inactive. Politely ask if they are still present. "
    "Use one short sentence."
)
_console_test_lead: dict[str, object] | None = None
CONSOLE_LEAD_OPTIONS = frozenset(
    {
        "--company-name",
        "--contact-name",
        "--city",
        "--state",
        "--industry",
        "--tier",
        "--phone",
        "--website-domain",
        "--lead-candidate-id",
        "--current-score",
        "--why-now-summary",
        "--naics-code",
    }
)

server = AgentServer(
    num_idle_processes=1,
    initialize_process_timeout=40.0,
)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = inference.VAD(
        model="silero",
        min_silence_duration=0.55,
    )


server.setup_fnc = prewarm


def _console_lead_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--company-name", default="Example Company")
    parser.add_argument("--contact-name", default="Alex")
    parser.add_argument("--city", default="Austin")
    parser.add_argument("--state", default="Texas")
    parser.add_argument("--industry", default="staffing")
    parser.add_argument("--tier", default="Warm")
    parser.add_argument("--phone")
    parser.add_argument("--website-domain", default="")
    parser.add_argument("--lead-candidate-id")
    parser.add_argument("--current-score", type=float)
    parser.add_argument("--why-now-summary")
    parser.add_argument("--naics-code")
    return parser


def default_console_lead() -> dict[str, object]:
    return vars(_console_lead_parser().parse_args([]))


def parse_console_lead_args(
    argv: list[str],
) -> tuple[list[str], dict[str, object] | None]:
    if len(argv) < 2 or argv[1] != "console":
        return argv, None

    parser = _console_lead_parser()
    values, livekit_args = parser.parse_known_args(argv[2:])
    lead_was_configured = any(
        argument.split("=", 1)[0] in CONSOLE_LEAD_OPTIONS for argument in argv[2:]
    )
    lead = vars(values) if lead_was_configured else None
    return [argv[0], "console", *livekit_args], lead


def select_lead_for_session(
    database_lead: dict[str, object] | None,
    *,
    is_console: bool,
    console_lead: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, bool]:
    if is_console and console_lead is not None:
        return dict(console_lead), False
    if database_lead is not None:
        return database_lead, True
    if is_console:
        return default_console_lead(), False
    return None, False


def build_turn_handling(settings: Settings) -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(
            version="v1",
            api_key=settings.livekit_inference_api_key,
            api_secret=settings.livekit_inference_api_secret,
        ),
        endpointing={
            "mode": "fixed",
            "min_delay": 0.3,
            "max_delay": 2.5,
        },
        interruption={
            "mode": "adaptive",
            "min_duration": 0.5,
            "min_words": 0,
            "false_interruption_timeout": 2.0,
            "resume_false_interruption": True,
        },
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": False,
        },
    )


def log_transcript(event: object, room_name: str, agent: Aiva) -> None:
    item = getattr(event, "item", None)
    if not isinstance(item, ChatMessage) or item.role not in {"user", "assistant"}:
        return
    text = item.text_content.strip()
    if text:
        logger.info(
            "transcript room=%r role=%s stage=%s text=%r",
            room_name,
            item.role,
            agent.current_stage,
            text,
        )


def install_inactivity_handler(
    session: AgentSession,
    agent: Aiva,
    room_name: str,
) -> None:
    inactivity_task: asyncio.Task[None] | None = None

    async def check_if_user_present() -> None:
        nonlocal inactivity_task
        try:
            for attempt in range(1, INACTIVITY_REMINDERS + 1):
                if agent.is_ending:
                    return
                logger.info(
                    "user_inactive_reminder room=%r attempt=%d",
                    room_name,
                    attempt,
                )
                speech = session.generate_reply(
                    instructions=INACTIVITY_INSTRUCTION,
                    allow_interruptions=True,
                )
                await speech.wait_for_playout()
                await asyncio.sleep(INACTIVITY_REMINDER_INTERVAL)

            if not agent.is_ending:
                logger.info("user_inactive_exhausted room=%r", room_name)
                inactivity_task = None
                await agent.end_for_inactivity()
        except asyncio.CancelledError:
            logger.info("user_inactive_cancelled room=%r", room_name)
        finally:
            if inactivity_task is asyncio.current_task():
                inactivity_task = None

    def cancel_inactivity() -> None:
        nonlocal inactivity_task
        task = inactivity_task
        inactivity_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def on_user_state_changed(event: UserStateChangedEvent) -> None:
        nonlocal inactivity_task
        if event.new_state == "away":
            if agent.is_ending or (inactivity_task is not None and not inactivity_task.done()):
                return
            logger.info("user_away room=%r", room_name)
            inactivity_task = asyncio.create_task(check_if_user_present())
            return

        if inactivity_task is not None:
            logger.info("user_resumed room=%r state=%s", room_name, event.new_state)
            cancel_inactivity()

    session.on("user_state_changed", on_user_state_changed)
    session.on("close", lambda _: cancel_inactivity())


@server.rtc_session(agent_name=os.getenv("AGENT_NAME", "codex-agent"))
async def entrypoint(ctx: JobContext) -> None:
    is_console = ctx.is_fake_job()
    settings = Settings.from_env(allow_livekit_cloud=is_console)
    repository = await PostgresRepository.connect()
    try:
        database_lead = await repository.get_next_lead()
    except Exception:
        await repository.close()
        raise
    lead, persist_call = select_lead_for_session(
        database_lead,
        is_console=is_console,
        console_lead=_console_test_lead,
    )
    if lead is None:
        logger.info("no eligible leads available")
        await repository.close()
        await ctx.delete_room()
        return
    if not persist_call:
        logger.info("console_test_lead_loaded phone_available=false")

    room_id = ctx.job.room.sid or ctx.room.sid or ctx.room.name
    call_id: int | None = None
    if persist_call:
        try:
            call_id = await repository.create_call(
                lead["lead_candidate_id"],
                settings.agent_version,
                "openai",
                "cartesia",
                room_id,
            )
        except Exception:
            await repository.close()
            raise
    context_variables = build_context_variables(lead)
    agent = Aiva(
        lead,
        context_variables,
        ctx,
        repository,
        test_mode=settings.test_mode or not persist_call,
    )
    stt = deepgram.STT(model="nova-2")
    llm = openai.LLM(model="gpt-4o-mini")
    tts = cartesia.TTS(
        voice="a33f7a4c-100f-41cf-a1fd-5822e8fc253f",
        speed=1.0,
    )
    vad = ctx.proc.userdata["vad"]
    turn_handling = build_turn_handling(settings)
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_handling=turn_handling,
        user_away_timeout=15.0,
    )
    install_inactivity_handler(session, agent, ctx.room.name)
    session.on(
        "conversation_item_added",
        lambda event: log_transcript(event, ctx.room.name, agent),
    )
    session.on(
        "error",
        lambda event: logger.error("agent_session_error error=%s", event.error),
    )

    finalized = False

    async def finalize(reason: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True
        objections = "; ".join(agent.objection_log) or None
        try:
            if call_id is not None:
                await repository.complete_call(
                    call_id,
                    lead["lead_candidate_id"],
                    agent.call_result,
                    referral_details=agent.referral_details,
                    callback_details=agent.callback_details,
                    objection_text=objections,
                    update_lead=not settings.test_mode,
                )
            logger.info(
                "call_finished room=%r result=%s reason=%r persisted=%s",
                ctx.room.name,
                agent.call_result,
                reason,
                call_id is not None,
            )
        finally:
            await repository.close()

    ctx.add_shutdown_callback(finalize)
    logger.info(
        "call_started room=%r company=%r tier=%r version=%s "
        "turn_detector=v1 interruption=adaptive",
        ctx.room.name,
        lead["company_name"],
        lead["tier"],
        settings.agent_version,
    )
    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(delete_room_on_close=False),
        )
        await ctx.connect()
    except Exception:
        await finalize("entrypoint_error")
        raise


def main() -> None:
    global _console_test_lead
    sys.argv, _console_test_lead = parse_console_lead_args(sys.argv)
    if sys.argv[1:2] != ["console"]:
        Settings.from_env()
    cli.run_app(server)


if __name__ == "__main__":
    main()
