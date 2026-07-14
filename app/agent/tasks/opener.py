"""Opener task — company confirm, right-person check, AI disclosure."""

import asyncio
import os

from livekit.agents import AgentTask, function_tool

from app.agent.speech import NATURAL_SPEECH_GUIDE, SpeechSafetyMixin


# ============================================================
# TASK 1 — OPENER
# ============================================================

# How long Aiva waits for the prospect to respond to each spoken "Hello?"
# before escalating to the next attempt. She always speaks first — real
# phone etiquette on an outbound call means the caller opens, never the
# person who picked up — so this only governs the listen window after each
# of her own attempts, never a wait for their unprompted greeting. No native
# LiveKit primitive covers "wait for the next user turn, with a timeout" —
# AgentSession's user_away_timeout is a whole-call inactivity flag (resets on
# any speech, defaults to 15s), not a one-shot per-attempt race — so this is
# a manual asyncio.Event raced against asyncio.wait_for() in
# OpenerTask.on_enter(). Kept tunable via .env since the right value can only
# be judged from real Twilio call latency (SIP pickup + STT/VAD lag), not the
# browser playground.
OPENER_WAIT_TIMEOUT_SECONDS = float(os.getenv("OPENER_WAIT_TIMEOUT_SECONDS", "3.5"))

# How many times Aiva speaks a "Hello?" into dead air before giving up on the
# call entirely. Attempt 1 is unconditional — she always speaks it immediately
# on connecting, never waits silently first. Each attempt speaks, then races
# the same asyncio.Event/wait_for pattern above for another
# OPENER_WAIT_TIMEOUT_SECONDS window before escalating. Tone escalates attempt
# over attempt — neutral, then doubtful, then a final check — so it reads
# like a real person checking a quiet line instead of the same line repeated
# three times.
OPENER_MAX_HELLO_ATTEMPTS = int(os.getenv("OPENER_MAX_HELLO_ATTEMPTS", "3"))

# One entry per attempt, indexed 0..OPENER_MAX_HELLO_ATTEMPTS-1. If
# OPENER_MAX_HELLO_ATTEMPTS is tuned via .env, extend this list to match.
# These are pure "is anyone there" checks, never the TURN 1 company
# confirmation — that only starts once a response is actually heard.
_HELLO_ATTEMPT_INSTRUCTIONS = [
    # Attempt 1 — spoken unconditionally, the instant the call connects.
    # Neutral tone, matches a normal human answering the phone.
    'Open the call now. Say exactly a brief, neutral "Hello?" — nothing '
    "else. No company confirmation, no disclosure, no Porter Capital "
    "mention yet — just checking if anyone is on the line.",
    # Attempt 2 — doubtful, like checking whether the line is still connected.
    'Still no response. Say something short and doubtful that the line is '
    'still live — e.g. "Hello...? Still there?" One short line, rising '
    "uncertainty in tone. Nothing about the company or AI yet.",
    # Attempt 3 — final try, most doubtful/final-check tone.
    'One more silent stretch after two tries. Say one last short line, most '
    'doubtful — e.g. "Hello? Is anyone there?" One short line, then stop.',
]


class OpenerTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, lead):
        company = lead.get('company_name', 'your company')
        city    = lead.get('city', '')

        super().__init__(
            instructions=f"""
            You are Aiva, an AI sales assistant at Porter Capital.
            Calling: {company} in {city}

            {NATURAL_SPEECH_GUIDE}

            YOUR ONLY JOB: Open this call like a real human phone rep would
            — confirm you've reached the right company FIRST, and only
            disclose the AI/cold-call framing and ask for time AFTER that's
            confirmed. Never do both in the same breath.

            TURN 1 — COMPANY CONFIRMATION ONLY. Ask exactly this, nothing
            more (no disclosure, no Porter Capital mention yet):
            "Hey — is this someone over at {company}?"

            Listen to their answer and branch:
            - CONFIRMED (yes, this is {company}, or anything affirming it)
              -> move to the RIGHT-PERSON CHECK below, in your next reply.
              Do not call task_complete yet.
            - CONFUSED (they didn't catch it — "who's this?", "what
              company?", "sorry, what?") -> briefly repeat or rephrase the
              same confirmation question ONE time, naturally. Do not
              proceed to TURN 2 until they've actually confirmed or denied
              the company.
            - ANY NON-CONFIRMATION (this is TURN 1 ONLY — a plain "no," "wrong
              number," "no one here by that name," or still unclear after one
              rephrase — every one of these means the same thing here: not
              yet a confirmed wrong number, just an unconfirmed one) -> do
              NOT call task_complete yet and do NOT apologize yet. Ask
              exactly ONE soft follow-up first, naturally, to rule out a
              mishearing or a subsidiary/different-location name mismatch:
              "Ah — are you with {company} at all, or is this a different
               business?"
              This is TURN 1B. Listen to their answer and branch:
              - They confirm they ARE with {company} after all (e.g. they
                misheard, or answered under a different location/subsidiary
                name) -> this is NOT a wrong number. Move to the
                RIGHT-PERSON CHECK below, in your next reply, exactly as if
                they'd confirmed the first time. Do not call task_complete
                yet.
              - They confirm they are NOT with {company} (still no, wrong
                business, no one by that name) -> now it's a confirmed
                wrong number. Call the task_complete tool with result set
                to wrong_number, and do not speak anything yourself in this
                turn — no apology, no goodbye, no other words. The
                task_complete tool speaks the apology itself, in a
                separate, guaranteed step, before the call ends; anything
                you say here would either double up on it or get cut off
                by the tool firing. Do not attempt TURN 2 or the pitch.
              This is a dialing/data problem, not a sales decision — there
              is NO bad_timing/not_interested clarifying question at TURN 1,
              ever, no matter how the "no" is phrased. That clarifying
              question in the TURN 2 rules below does not exist yet at this
              point in the call — do not reach for it here.

            RIGHT-PERSON CHECK — once the company is confirmed (either
            directly at TURN 1 or via TURN 1B), before disclosing anything
            about AI or asking for time, confirm you've got the right
            person. Ask exactly this:
            "I'm calling about invoice funding for {company} — are you the
             right person for that, or is there someone else who handles
             it?"

            Listen to their answer and branch:
            - CONFIRMS they ARE the right person (e.g. "yes, that's me,"
              "I handle that," or any other affirming answer) -> move to
              TURN 2 below, in your next reply. Do not call task_complete
              yet.
            - Says they are NOT the right person -> ask exactly this:
              "No problem — who would be the right person, and is there a
               good way I could reach them?"
              Listen to their answer and branch:
              - They provide a name and/or a way to reach that person
                (phone number, email, extension, department, or something
                like "just call back and ask for accounting") -> acknowledge
                warmly and thank them for the info in one short line — e.g.
                "Got it, thanks so much — have a great day!" — then call the
                task_complete tool with result set to gatekeeper_referral and
                referral_details set to exactly what they told you (the
                name/contact info/department, verbatim or close to it). Do
                not attempt TURN 2 or the pitch.
              - They don't know who the right person is, or say no one else
                is available or relevant -> call the task_complete tool with
                result set to not_interested. Do not speak a closing line
                yourself here — the tool's caller handles the polite close.
                Do not attempt TURN 2 or the pitch.
              This is a routing dead end, not a sales objection — there is
              no bad_timing distinction here, only whether they gave you a
              referral or not.
            - UNCLEAR (doesn't clearly confirm or deny — a mishearing, an
              unrelated remark, background noise transcribed as words,
              anything that isn't unambiguously one of the two branches
              above) -> do NOT call task_complete and do NOT guess which
              branch it is. Ask exactly ONE clarifying rephrase first,
              naturally: "Sorry, just to confirm — are you the right person
              to talk to about that, or should I ask for someone else?"
              - If their answer to the rephrase is now clear, follow the
                CONFIRMS or NOT branch above accordingly.
              - If it's still unclear after that one rephrase, treat it
                conservatively as NOT the right person and follow that
                branch. Never call task_complete with result hung_up here
                as a fallback for unclear or ambiguous speech — hung_up is
                reserved for when the prospect has actually gone silent or
                disconnected, not for input you're unsure how to route.

            TURN 2 — once the right person is confirmed, disclose and ask
            for time, combined in one short reply. Use this closely:
            "Hey, so I'll be upfront — this is actually a cold call, and
             I'm an AI, Aiva, calling for Porter Capital. Got 30 seconds?
             Totally fine to hang up too if now's not good."

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            RULES:
            - Never disclose AI or mention Porter Capital during TURN 1 or
              the RIGHT-PERSON CHECK — those are confirmation-only. "Porter
              Capital" is never said until TURN 2.
            - Under 2 sentences per turn.
            - Porter Capital is who Aiva works for — always say "Porter
              Capital" naturally in TURN 2. Never genericize it into
              something like "a financial services company."
            - The PROSPECT'S/LEAD'S company name ({company}) is only for
              the TURN 1 confirmation question and the RIGHT-PERSON CHECK
              question — never say it again after that, it's internal
              context otherwise.
            - After they respond to TURN 2, call the task_complete tool —
              do not write it out as text. The tool call itself is a
              separate, silent action, never part of what you say out loud.
            - THIS RULE APPLIES AT TURN 2 ONLY, never at TURN 1 (TURN 1's
              wrong_number handling above is separate and already complete
              in itself — never apply anything below to TURN 1). At TURN 2,
              if the prospect says anything indicating they want to end the
              call, never push back or ask "are you sure."
              Only skip straight to a result WITHOUT the clarifying question
              below when their wording is UNAMBIGUOUSLY final — e.g. "not
              interested," "don't call here again," "take me off your
              list," "we're all set," "no thank you, we don't need that" —
              or they've already hung up (hung_up).
              For anything shorter or less clear-cut — including a bare
              "no," "nope," or "not right now" — do NOT guess which one it
              is. Ask exactly ONE soft clarifying question first — this is
              not pushing back, it's giving them an easy out:
              "Totally fair. Can I ask real quick — is it bad timing, or just
               not something you need right now?"
              Then listen to which one it is — this single answer decides the
              result you pass:
              - If it's a timing thing ("busy right now", "call me later",
                "not a good time", "maybe down the road") -> that's bad_timing.
              - If it's a genuine no ("just don't need it", "we're all set",
                "not for us") -> that's not_interested.
            - If they agree to keep listening, that routes to a short pitch
              next — NOT straight to qualifying questions.
            """
        )
        # Gates task_complete() below: True for as long as on_enter's
        # hello-escalation loop hasn't heard anything yet. There is no real
        # conversation during that phase, so a stray task_complete call
        # (e.g. from the SDK's own turn-completion auto-reply firing on
        # noise/backchannel before heard_speech is set) is never correct
        # and must be a no-op until the loop itself clears this.
        self._hello_phase_active = True

    async def on_enter(self):
        heard_speech = asyncio.Event()

        def _on_user_state(ev):
            # "speaking" fires on VAD speech-onset, well before STT finalizes
            # anything — the earliest possible "they said something" signal,
            # which is what this race needs (not a completed transcript).
            if ev.new_state == "speaking":
                heard_speech.set()

        # Listener is armed/disarmed per-attempt, scoped to only the wait_for
        # window below — NOT across generate_reply's own TTS playback. Aiva's
        # own "Hello?" audio bleeding into the mic (echo) can flip VAD to
        # "speaking" while she's still talking; if the listener were live
        # during that playback, heard_speech would already be set by the time
        # wait_for is reached, so it'd return instantly on false ground and
        # this loop would silently exit after attempt 0 — no attempt 2, no
        # timeout, no hangup. See UPGRADE_NOTES.md for the known AEC
        # limitation this works around.
        # Aiva always speaks first — attempt 0 (spoken as "Hello?") is
        # unconditional, never a silent wait for their own greeting.
        for attempt in range(OPENER_MAX_HELLO_ATTEMPTS):
            if self.done():
                # task_complete() beat us to it (e.g. a race that slipped
                # through the _hello_phase_active gate) — nothing left
                # for this loop to do.
                return
            await self.session.generate_reply(
                instructions=_HELLO_ATTEMPT_INSTRUCTIONS[attempt]
            )
            heard_speech.clear()
            self.session.on("user_state_changed", _on_user_state)
            try:
                await asyncio.wait_for(
                    heard_speech.wait(), timeout=OPENER_WAIT_TIMEOUT_SECONDS
                )
                # They spoke — hand off to Turn 1. task_complete() is now
                # live; say nothing here. The default LiveKit
                # turn-completion flow (AgentActivity._user_turn_completed_task)
                # auto-generates the reply from these instructions once
                # their turn finishes, so Aiva responds to their actual
                # reply naturally instead of talking over it.
                self._hello_phase_active = False
                return
            except asyncio.TimeoutError:
                if attempt >= OPENER_MAX_HELLO_ATTEMPTS - 1:
                    # Silent through every spoken attempt — give up.
                    # Shares the 'hung_up' teardown path (no closing
                    # words possible) via start_call() -> _end_call()
                    # with no closing_instructions. call_result there is
                    # already set to 'no_answer', not 'hung_up' — nobody
                    # ever spoke at all, so 'no_answer' is the accurate
                    # status, distinct from a real hung_up where someone
                    # was there and disconnected.
                    self._hello_phase_active = False
                    if not self.done():
                        self.complete('hung_up')
                    return
            finally:
                self.session.off("user_state_changed", _on_user_state)

    @function_tool()
    async def task_complete(self, result: str, referral_details: str = ""):
        """Call when prospect responds.
        result: interested / not_interested / bad_timing / wrong_number /
        hung_up / gatekeeper_referral.
        referral_details: the name/contact info they gave for the right
        person, only when result is gatekeeper_referral."""
        if self._hello_phase_active:
            # on_enter's hello-escalation loop hasn't heard heard_speech yet
            # — there is no real conversation to react to. This guards
            # against the SDK's own turn-completion auto-reply (see
            # AgentActivity._user_turn_completed_task) firing a reply from
            # this task's instructions off VAD noise/backchannel before the
            # loop has actually handed off to Turn 1, and the LLM using that
            # stray turn to call this tool early. on_enter alone owns
            # completion during this phase.
            print(f"[OpenerTask] task_complete('{result}') ignored — still in hello-escalation phase")
            return "Not yet — still waiting for the prospect to respond to the greeting."
        if self.done():
            # Lost the race with on_enter's own give-up path, or called
            # twice — already-done is an expected outcome here, not an
            # error, so this is a silent no-op rather than letting the
            # SDK's RuntimeError propagate.
            print(f"[OpenerTask] task_complete('{result}') ignored — task already complete")
            return "Call already completed."
        if result == 'gatekeeper_referral':
            self.complete({'result': result, 'referral_details': referral_details})
            return
        if result == 'wrong_number':
            # Tool execution runs concurrently with (and finishes far faster
            # than) TTS playout of any text the LLM generated in this same
            # turn — instructing the model to "apologize, then call
            # task_complete" in prose does not make the apology finish
            # playing before self.complete() below tears the call down via
            # PorterSupervisor._end_call()'s delete_room(). generate_reply()
            # awaits its SpeechHandle, which the SDK resolves only once
            # playout is actually complete, so awaiting it HERE — inside the
            # tool call the SDK already awaits before treating this turn as
            # done — is what actually guarantees ordering, not the prompt.
            await self.session.generate_reply(
                instructions='Apologize briefly for the mix-up — e.g. '
                '"Oh — sorry about that, my mistake." One short sentence, '
                'then stop.',
                allow_interruptions=False,
            )
        if not self.done():
            self.complete(result)


