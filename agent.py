# ============================================================
# agent.py — Porter Capital Voice Agent v3
# Agent name: Aiva
# ============================================================

import os
import re
import time
import json
import asyncio
import urllib.request
from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    Agent,
    AgentTask,
    RoomInputOptions,
    cli,
    WorkerOptions,
    JobExecutorType,
    JobProcess,
    function_tool,
    JobContext,
)
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import groq, cartesia, deepgram, silero, anthropic, openai as openai_plugin, elevenlabs
from db import get_next_lead, update_lead_status, add_to_suppression
from knowledge import PORTER_CAPITAL_KNOWLEDGE

load_dotenv()

current_lead = None


# ============================================================
# NATURAL SPEECH GUIDE — applied to every task
# ============================================================
# This is injected into every task so the tone is consistent.
# Think of it as coaching the agent on HOW to speak, not WHAT to say.

NATURAL_SPEECH_GUIDE = """
SPEAK LIKE A SHARP, CONFIDENT HUMAN SALES REP.

The most important rules:

1. SHORT RESPONSES ONLY
   Maximum 2 sentences per response.
   One sentence is better.
   Never stack multiple thoughts.

2. ANSWER BEFORE ASKING
   Always address what they said first.
   Then ask ONE follow-up question if needed.
   Never ask a question without answering first.

3. NEVER REPEAT YOURSELF
   If you said something already, do not say it again.
   Move the conversation forward every single turn.

4. CONTRACTIONS ALWAYS
   I'm, we're, you're, don't, it's, I'll, that's
   Never: I am, we are, you are, do not

5. NATURAL CONNECTORS
   Start responses with: Yeah, Right, Got it, Oh, So, Sure
   Never start with: Absolutely, Certainly, Of course,
   That's great, Hello, I understand

6. BANNED WORDS — NEVER USE THESE:
   Absolutely, Certainly, Wonderful, Leverage, Solutions,
   Utilize, Transparent, Outstanding, Facilitate, Endeavor

7. NO UNNECESSARY QUESTIONS
   Only ask a question when you genuinely need information
   to move the call forward.
   Never ask questions just to fill silence.

8. WRITE PAUSES AND EMPHASIS DIRECTLY INTO YOUR TEXT:
   TTS reads exactly what it's given, so pacing has to be written in.
   - Use "..." for a natural pause or hesitation
     Example: "So, honestly... it kind of depends."
   - Use em-dashes for a small break in thought
     Example: "We work with — pretty much any B2B business."
   - Occasionally trail off naturally instead of finishing formally
     Example: "Rates run about 1 to 5 percent, but yeah — depends
     on the specifics."
   - Break up long sentences into two shorter ones with a pause
     between them rather than one long flowing sentence
   - Do NOT use exclamation points — they read as fake enthusiasm
   - A slight, natural stumble occasionally reads as more human than
     a perfectly smooth sentence: "We — we can definitely look into
     that for you."
"""


# ============================================================
# SPEECH SAFETY NET
# ============================================================
# Backstop for when the prompt-only "never speak this" instructions
# don't hold and Groq narrates internal instructions/routing logic
# or a raw tool-call as if it were dialogue. Strips those lines from
# the LLM output before it ever reaches TTS, independent of whether
# the model followed the prompt.

# A leaked raw tool call — <function=...>{...}</function> — is the common
# failure and it almost always arrives INLINE, tacked onto the end of an
# otherwise-clean sentence ("...straightforward. <function=task_complete ...>").
# The old ^-anchored, whole-line filter never caught that. These strip the
# fragment WHEREVER it appears while keeping the surrounding dialogue:
#   - _FUNCTION_LEAK removes a well-formed <function ...>...</function> span.
#   - _FUNCTION_LEAK_UNCLOSED removes a malformed/unclosed variant (no closing
#     tag, truncated JSON) by consuming from <function to end-of-text.
# DOTALL so a fragment may span newlines; IGNORECASE for tag casing. Quoting
# inside the fragment is irrelevant — we never parse it, we delete it.
_FUNCTION_LEAK          = re.compile(r'<function\b.*?</function\s*>', re.IGNORECASE | re.DOTALL)
_FUNCTION_LEAK_UNCLOSED = re.compile(r'<function\b.*', re.IGNORECASE | re.DOTALL)

# Once a tacked-on fragment is removed, the sentence often ends on the pause
# marker that led into it ("...callback details..." / "Sounds good —"). Strip a
# trailing run of 2+ dots, an ellipsis char, or a dangling connector, but leave
# a single sentence terminator (. ? !) intact so real sentences stay complete.
_DANGLING_TAIL = re.compile(r'\s*(?:\.{2,}|[…\-—:;,]+)\s*$')

# Whole-line meta-instruction leaks (routing prose, name/parameters JSON) that
# are never dialogue in the first place — still filtered line-by-line.
_LEAK_LINE_PATTERNS = [
    re.compile(r'(?im)^\s*(if|when)\s+(they|the\s+(prospect|caller|lead))\s+(say|says|ask|asks)\b.*:\s*$'),
    re.compile(r'(?im)^\s*\{?"?name"?\s*:\s*"?\w+".*\bparameters\b.*'),
]


def strip_leaked_meta_text(text: str) -> str:
    """Strip leaked raw tool calls and meta-instruction lines from spoken text.

    Removes inline <function ...> fragments (closed or malformed) wherever they
    appear, drops whole-line meta-instruction leaks, and tidies any pause
    punctuation left dangling once an end-of-sentence fragment is removed.
    """
    # 1. Excise inline tool-call fragments (well-formed first, then any
    #    unclosed remainder running to end-of-text).
    text = _FUNCTION_LEAK.sub("", text)
    text = _FUNCTION_LEAK_UNCLOSED.sub("", text)

    # 2. Drop whole-line meta-instruction leaks.
    lines = text.split("\n")
    lines = [ln for ln in lines if not any(p.search(ln) for p in _LEAK_LINE_PATTERNS)]
    text = "\n".join(lines)

    # 3. Clean up dangling pause punctuation at the very end.
    text = _DANGLING_TAIL.sub("", text.rstrip())
    return text.strip()


class SpeechSafetyMixin:
    """Mixed into every Agent/AgentTask so its spoken output passes through
    strip_leaked_meta_text before synthesis, regardless of which task is active."""

    async def tts_node(self, text, model_settings):
        async def _filtered():
            full_text = "".join([chunk async for chunk in text])
            cleaned = strip_leaked_meta_text(full_text)
            if cleaned:
                yield cleaned

        return Agent.default.tts_node(self, _filtered(), model_settings)


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


# ============================================================
# TASK 2 — PITCH
# ============================================================

class PitchTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, lead):
        super().__init__(
            instructions=f"""
            You are Aiva. The prospect just agreed to listen (said yes to
            the ice breaker).

            You are mid-call. The prospect already heard the ice breaker and
            AI disclosure from earlier in this conversation. Never repeat
            the opening greeting, never re-introduce yourself, never
            re-disclose being an AI unless directly asked again. Jump
            straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            YOUR ONLY JOB: deliver the pitch below. USE THIS SCRIPT CLOSELY
            — it is locked-in and approved, not a loose illustration. Do
            not substitute your own generic paraphrase of it. You may only
            adapt it for natural delivery (contractions, minor word
            choice, pause placement) — never drop "Porter Capital," never
            drop the differentiator, and never change the closing question
            into anything else.

            The script covers: who you help, what result you deliver, why
            it matters, plus ONE differentiator, then a soft check-in
            question — NOT a qualifying question. Cut straight from their
            check-in response into the pitch content — no detour question
            first.

            Pull ONE of these differentiators naturally, whichever fits the
            moment (never all three, never recited as a list):
            - "We've been doing this for decades, funded billions to
              businesses nationwide."
            - "People here can make decisions fast, not stuck waiting on
              some committee."
            - "You'd get a dedicated person who actually knows your
              business, not a random rep in a queue."

            SCRIPT (use this closely, do not write your own version):
            "So — we're Porter Capital... we help businesses like yours get
             paid faster on outstanding invoices, instead of waiting weeks
             or months. We've actually been doing this for decades — funded
             billions to businesses out there. And unlike a lot of places,
             you'd get a real person who can make a decision fast, not stuck
             waiting on some committee. Does that sound worth a quick chat?"

            The closing line is always some version of "does that sound
            worth a quick chat?" — a soft check-in, never "what's your
            current process for X" or any other fact-finding question.
            Ending the pitch on a qualifying-style question is a rule
            violation, not an acceptable variation.

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            RULES:
            - Porter Capital is who Aiva works for — always say "Porter
              Capital" naturally if you reference who you're calling from.
              Never genericize it into something like "a financial
              services company."
            - The PROSPECT'S/LEAD'S company name is different — that one is
              internal context only, never spoken aloud to them.
            - Keep it tight and conversational. No rambling.
            - Never ask about their role, title, or job function — that
              doesn't serve qualification and just adds an unnecessary turn.
              Go straight from their check-in response into the pitch.
            - The check-in question is soft ("worth a quick chat?"),
              never a qualifying question ("what type of business are you in?").
            - NEVER quote rates, percentages, or dollar amounts.

            When they respond, call the task_complete tool — do not write
            it out as text. Speak only the pitch and the check-in question;
            the tool call itself is a separate, silent action, never part
            of what you say out loud.

            [INTERNAL ROUTING LOGIC — never speak any of this, it only
            controls which tool arguments you pass]: if they respond
            positively, finish by calling the tool with result set to
            interested. If they show resistance or an objection, finish by
            calling the tool with result set to objection and put what they
            said in objection_text.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Deliver the short pitch now, including one natural differentiator. Keep it tight and conversational, then ask the soft check-in question."
        )

    @function_tool()
    async def task_complete(self, result: str, objection_text: str = ""):
        """result: interested / objection. objection_text: what they said, if result is objection."""
        if result == "objection":
            self.complete({"result": result, "objection_text": objection_text})
        else:
            self.complete({"result": result, "objection_text": ""})


# ============================================================
# TASK 3 — QUALIFIER
# ============================================================

class QualifierTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, lead):
        company = lead.get('company_name', 'your company')

        super().__init__(
            instructions=f"""
            You are Aiva, an AI sales assistant at Porter Capital.
            You are speaking with someone at {company}.

            You are mid-call. The prospect already heard the ice breaker and
            AI disclosure from earlier in this conversation. Never repeat
            the opening greeting, never re-introduce yourself, never
            re-disclose being an AI unless directly asked again. Jump
            straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            YOUR ONLY JOB: Qualify this prospect naturally.
            Find out what type of business they are, and whether they
            factor invoices.

            Ask ONE question at a time. Acknowledge their answer first.

            Your qualifying question is: "What type of business are you in?"
            Then react naturally to whatever they say — there's no fixed
            script for the follow-up, just acknowledge their answer and
            ask if they currently factor invoices.

            Follow-up template: "Got it — [natural reaction to their
            answer]. And do you guys currently factor any of your
            invoices, or handle that a different way?"

            OFF-SCRIPT QUESTIONS (process, eligibility, rates):
            An engaged prospect asking questions about the process,
            eligibility, or rates is NOT a disqualifying signal — it's
            often a sign of genuine interest. Answer briefly, defer
            specifics to the human advisor, then return to qualifying or
            move toward booking if they seem ready. Do NOT call
            task_complete just because the conversation went off-script —
            only call it once you've actually determined qualified or
            not_qualified based on business type and factoring status (or
            a clear negative signal from the prospect).

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            CRITICAL CONTEXT RULE:
            This is an OUTBOUND cold call. WE called THEM.
            Never ask "what prompted you to look at other options"
            or any question implying they reached out to us.
            We called them. They did not call us.

            [INTERNAL REASONING — never speak this guidance, only the actual
            reply text below it]: once they signal they're open to
            exploring, move directly toward booking — do not ask why they
            are open, just move forward with something in your own words
            like:
            "Yeah? Cool — so our advisor can walk you through
             exactly how it works for a business like yours.
             Want me to set up a quick call with them?"

            HARD RULES:
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - NEVER quote rates, percentages, or dollar amounts
            - If asked about rates, say only:
              "Honestly it depends on your volume — our advisor
               gets you an exact number in 15 minutes."
            - One question at a time. Always.

            Use this knowledge to answer questions accurately:
            {PORTER_CAPITAL_KNOWLEDGE}

            When you have enough info, call the task_complete tool — do
            not write it out as text. Speak only your natural reply; the
            tool call itself is a separate, silent action, never part of
            what you say out loud.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Start qualifying naturally. Acknowledge what they said first if they said anything."
        )

    @function_tool()
    async def task_complete(self, result: str):
        """result: qualified / not_qualified"""
        self.complete(result)


# ============================================================
# TASK 4 — OBJECTION HANDLER
# ============================================================

class ObjectionTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, objection: str):
        super().__init__(
            instructions=f"""
            You are Aiva. The prospect just said: "{objection}"

            You are mid-call. The prospect already heard the ice breaker and
            AI disclosure from earlier in this conversation. Never repeat
            the opening greeting, never re-introduce yourself, never
            re-disclose being an AI unless directly asked again. Jump
            straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            Follow this formula STRICTLY: Acknowledge -> Ask a question -> Continue.
            Never defend, never argue, never over-explain. One acknowledgment,
            one question, then stop.

            Examples:
            "We already have a factor" ->
            "That makes sense... most people do. Just curious though — are
             you totally happy with what you're getting from them?"

            "Not interested" ->
            "Totally fair. Can I ask real quick — is it bad timing, or just
             not something you need right now?"
            Then listen to which one it is — this single answer decides the
            result you pass:
            - If it's a timing thing ("busy right now", "call me later",
              "not a good time", "maybe down the road") -> that's bad_timing.
            - If it's a genuine no ("just don't need it", "we're all set",
              "not for us") -> that's not_interested.

            "What are your rates" ->
            "Honestly, it depends on a few things — how much you're
             invoicing, your customers, stuff like that. Most places run
             somewhere between 1 and 5 percent... but we'd build you an
             actual number once we know more."

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            RULES:
            - Acknowledge first, in one short phrase. Never fight the objection.
            - Ask exactly ONE question. Never stack a question with an
              explanation, justification, or pitch.
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - ONE response then stop. Never push twice.
            - If they say no again, accept it gracefully and warmly.
            - NEVER quote rates or percentages. Ever.
            - Then call the task_complete tool — do not write it out as
              text. Speak only your acknowledgment and question; the tool
              call itself is a separate, silent action, never part of
              what you say out loud.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Acknowledge their objection in one short phrase, then ask exactly one question. Nothing else."
        )

    @function_tool()
    async def task_complete(self, result: str):
        """result: still_interested / not_interested / bad_timing / wants_callback"""
        self.complete(result)


# ============================================================
# TASK 5 — BOOKING
# ============================================================

class BookingTask(SpeechSafetyMixin, AgentTask):

    def __init__(self):
        super().__init__(
            instructions="""
            You are Aiva, an AI sales assistant at Porter Capital.
            The prospect has agreed to speak with a human advisor.

            You are mid-call. The prospect already heard the ice breaker and
            AI disclosure from earlier in this conversation. Never repeat
            the opening greeting, never re-introduce yourself, never
            re-disclose being an AI unless directly asked again. Jump
            straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            YOUR ONLY JOB: Get their callback details naturally.

            Do not make this feel like filling out a form.
            Sound genuinely happy they said yes.

            Open with something like:
            "Cool — let's get you set up with one of our advisors. They can
             actually dig into the numbers with you. What's a good number
             and time to grab you?"

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            Then confirm naturally:
            1. Best number to reach them
               (may differ from the one you called)
            2. Best day and time for a 15-minute call

            Confirm back naturally, then call the task_complete tool — do
            not write it out as text. Speak only your natural confirmation;
            the tool call itself is a separate, silent action, never part
            of what you say out loud.

            RULES:
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - Never quote rates or promise approval
            - Keep it warm and brief — they already said yes
            """.replace("{NATURAL_SPEECH_GUIDE}", NATURAL_SPEECH_GUIDE)
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="React warmly to their agreement. Ask for callback preference naturally, using the soft closing line about getting them set up with an advisor."
        )

    @function_tool()
    async def task_complete(self, callback_details: str):
        """Call when callback details are confirmed."""
        self.complete(f"booked:{callback_details}")


# ============================================================
# TASK 6 — DISCLOSURE
# ============================================================

class DisclosureTask(SpeechSafetyMixin, AgentTask):

    def __init__(self):
        super().__init__(
            instructions=f"""
            You are Aiva. Someone just asked if you are human or AI.

            You are mid-call. The prospect already heard the ice breaker
            earlier in this conversation. Never repeat the opening greeting
            or re-introduce yourself — just answer the question they just
            asked and jump straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            Answer like this — make it your own version:
            "Yeah, I am — an AI, just being real about it. Happy to keep
             chatting, or if you'd rather talk to one of our humans, I can
             set that up too."

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            RULES:
            - Never say you are human. Ever.
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - Sound unbothered and warm about being an AI
            - Do not apologize for being an AI
            - Then call the task_complete tool — do not write it out as
              text. Speak only your natural reply; the tool call itself is
              a separate, silent action, never part of what you say out
              loud.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Answer honestly. Be warm and unbothered about being an AI."
        )

    @function_tool()
    async def task_complete(self, prospect_reaction: str):
        """result: accepted / wants_human / wants_to_end"""
        self.complete(prospect_reaction)


# ============================================================
# TASK 7 — EXIT
# ============================================================

class ExitTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, lead):
        self.lead = lead
        super().__init__(
            instructions=f"""
            You are Aiva, an AI sales assistant at Porter Capital.
            The prospect wants to be removed from our call list.

            {NATURAL_SPEECH_GUIDE}

            YOUR ONLY JOB: Acknowledge warmly and end the call.

            Sound genuine — not robotic or over-apologetic.

            Example:
            "Of course — I'll take care of that right now. Thanks for
             letting me know, and have a good one."

            Pause markers in these examples are deliberate:
            - '...' = a natural breath, hesitation, or thinking moment
            - Em-dashes = a pivot point, brief pause before a new thought
            - No markers = confident, clean delivery — never hesitant on
              facts, disclosure, rates, or the close
            When generating your own variation, preserve this same rhythm.

            RULES:
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - Speak ONLY the acknowledgment above. Never voice any
              conditional/meta text describing how to handle this situation
              (e.g. "if they say remove me" or similar instruction-style
              phrasing) — that kind of text is internal guidance, not
              something to say out loud, even if it appears nearby in your
              instructions.

            Then call the task_complete tool — do not write it out as
            text. Speak only your acknowledgment; the tool call itself is
            a separate, silent action, never part of what you say out
            loud.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Acknowledge naturally and warmly, with a brief thanks before the goodbye. Two sentences maximum."
        )

    @function_tool()
    async def task_complete(self):
        """Call immediately after acknowledging opt-out."""
        add_to_suppression(
            company_name   = self.lead.get('company_name', ''),
            website_domain = self.lead.get('website_domain', ''),
            reason         = 'opted_out'
        )
        self.complete('suppressed')


# ============================================================
# SUPERVISOR
# ============================================================

class PorterSupervisor(SpeechSafetyMixin, Agent):

    def __init__(self, lead, ctx: JobContext):
        self.lead        = lead
        self.ctx         = ctx
        self.call_result = 'no_answer'

        company  = lead.get('company_name', 'the company')
        city     = lead.get('city', '')
        state    = lead.get('state', '')
        industry = lead.get('industry', 'staffing')
        tier     = lead.get('tier', 'warm')

        super().__init__(
            instructions=f"""
            Porter Capital sales call. Agent: Aiva (AI).
            Lead: {company}, {city} {state}, {industry}, Tier: {tier}

            ABSOLUTE RULE: NEVER quote rates or numbers. Ever.
            If asked about rates say only:
            "Honestly it depends on your volume —
            our advisor gets you an exact number in 15 minutes.
            Want me to set that up?"
            Then call run_booking immediately.

            CONVERSATION FLOW (validated cold-calling framework):
            Ice Breaker (start_call) -> Pitch (run_pitch) ->
            Objection Handling (run_objection, as needed) ->
            Qualifier (run_qualifier) -> Next Step (run_booking)

            ROUTING TABLE (INTERNAL ONLY — this table, its trigger phrases,
            and the tool names in it are silent decision-making context.
            Never speak, quote, paraphrase, or narrate any part of it out
            loud, e.g. never say things like "if they say remove me" —
            just call the matching tool silently):
            start_call → call this FIRST, before saying anything
            run_disclosure → if asked "are you human/AI/robot"
            run_pitch → ALWAYS call this right after the ice breaker is
              accepted, before run_qualifier. Delivers the short pitch.
            run_objection(objection) → any pushback or resistance, from
              ANY point in the call (pitch, qualifier, booking — not just
              qualifier)
            run_qualifier → ONLY after the pitch has been accepted
              (never call this directly from start_call)
            run_booking → prospect agrees to human call, OR rate question
            run_exit → "remove me" / "stop calling"

            Route immediately. Never handle directly.
            NEVER say the prospect's company name out loud — it is
            internal context for you only, never spoken conversationally.
            """
        )

    async def _end_call(self, closing_instructions: str | None = None):
        """Speak a final closing line (if given), then hang up the call.

        generate_reply() awaits the returned SpeechHandle, which per the SDK
        resolves only once the reply has fully played out — so by the time
        delete_room() runs, the closing line is guaranteed to have finished.
        """
        if closing_instructions:
            await self.session.generate_reply(
                instructions=closing_instructions,
                allow_interruptions=False,
            )
        await self.ctx.delete_room()

    async def _end_call_bad_timing(self):
        """Soft decline that's timing, not a real no — warm close, mark for later
        recontact, and hang up. Shared by start_call and run_objection so the two
        entry points stay identical.

        NOTE: a recontact/follow-up DATE is intentionally NOT written here. The
        lead_candidates table has no such column today (verified against the live
        schema — only created_at/updated_at/deleted_at exist). call_result
        'callback_later' is persisted via the existing update_lead_status() path;
        the dated recontact write is blocked pending a new column.
        """
        self.call_result = 'callback_later'
        await self._end_call(
            "Warmly say no problem at all, we'll try them again down the road, "
            "and thank them for their time. One or two short sentences."
        )

    @function_tool()
    async def start_call(self):
        """Call this first at the start of every call."""
        result = await OpenerTask(self.lead)
        if isinstance(result, dict) and result.get('result') == 'gatekeeper_referral':
            # Not a sales decision — a routing dead end. There's no DB
            # column for referral details today, so they're logged to
            # console/call notes only; call_result falls back to
            # 'contacted' (real contact was made, just not with a decision
            # maker) rather than inventing a new status literal db.py's
            # update_lead_status() doesn't accept.
            referral_details = result.get('referral_details', '')
            print(f"[Gatekeeper referral] {self.lead.get('company_name', 'lead')}: {referral_details}")
            self.call_result = 'contacted'
            await self._end_call(
                "Acknowledge warmly, thank them for the info, and end the "
                'call in one short sentence — e.g. "Got it, thanks so much '
                '— have a great day!"'
            )
            return "Gatekeeper referral provided. Call ended."
        if result == 'wrong_number':
            # TURN 1 outcome — a dialing/data problem, never a sales
            # decision. OpenerTask.task_complete() already awaited the
            # apology's playout internally before resolving, so no
            # additional closing line here — same "already said what
            # needed saying" pattern as hung_up.
            # Persisted as not_interested since there's no dedicated DB
            # status/column for a bad number today; the distinction lives
            # in this branch's behavior, not in what gets written to the DB.
            self.call_result = 'not_interested'
            await self._end_call()
            return "Wrong number at company confirmation. Call ended."
        if result == 'not_interested':
            self.call_result = 'not_interested'
            await self._end_call(
                "Acknowledge warmly that's fine, and end the call in one short sentence."
            )
            return "Prospect declined. Call ended."
        if result == 'bad_timing':
            await self._end_call_bad_timing()
            return "Bad timing at open. Marked callback_later. Call ended."
        if result == 'hung_up':
            # They already dropped the line — no closing words are possible,
            # so just tear the room down. Do NOT fall through to 'contacted'.
            self.call_result = 'no_answer'
            await self._end_call()
            return "Prospect hung up. Call ended."
        self.call_result = 'contacted'
        return f"Opening done. Result: {result}. Next: if positive, call run_pitch (not run_qualifier)."

    @function_tool()
    async def run_disclosure(self):
        """Prospect asked if Aiva is human or AI."""
        result = await DisclosureTask()
        if result == 'wants_human':
            # Route to booking in code — do not hand the decision back to the
            # LLM (same internal-delegation pattern run_pitch uses).
            return await self.run_booking()
        if result == 'wants_to_end':
            # Treat exactly like a decline: warm one-line close, then hang up.
            self.call_result = 'not_interested'
            await self._end_call(
                "Acknowledge warmly that's fine, thank them, and end the call in one short sentence."
            )
            return "Prospect wants to end after disclosure. Call ended."
        return f"Disclosure done. Result: {result}"

    @function_tool()
    async def run_pitch(self):
        """Prospect agreed to listen after the ice breaker. Deliver the short pitch."""
        result = await PitchTask(self.lead)
        if result['result'] == 'objection':
            return await self.run_objection(result['objection_text'])
        return "Pitch accepted. Move to run_qualifier next."

    @function_tool()
    async def run_objection(self, objection: str):
        """Prospect raised objection or pushback, from any point in the call. objection = what they said."""
        result = await ObjectionTask(objection)
        if result == 'not_interested':
            self.call_result = 'not_interested'
            await self._end_call(
                "Acknowledge warmly, wish them well, and end the call in one short sentence."
            )
            return "Not interested. Call ended."
        if result == 'bad_timing':
            await self._end_call_bad_timing()
            return "Bad timing. Marked callback_later. Call ended."
        if result == 'wants_callback':
            # Route to booking in code — same internal-delegation pattern as
            # run_pitch -> run_objection; never return a string and hope.
            return await self.run_booking()
        return f"Objection handled. Result: {result}"

    @function_tool()
    async def run_qualifier(self):
        """Pitch has already been accepted. Qualify the prospect."""
        result = await QualifierTask(self.lead)
        if result == 'qualified':
            return "Qualified. Move toward booking."
        self.call_result = 'not_interested'
        await self._end_call(
            "Acknowledge warmly, thank them for their time, and end the call in one short sentence."
        )
        return "Not qualified. Call ended."

    @function_tool()
    async def run_booking(self):
        """Prospect agreed to speak with human advisor."""
        result = await BookingTask()
        self.call_result = 'callback_booked'
        callback_details = result.split("booked:", 1)[1] if result.startswith("booked:") else result
        await self._end_call(
            "Confirm warmly that they're all set, referencing these callback details "
            f"naturally: {callback_details}. Say one of our advisors will call them then. "
            "Thank them for their time and say goodbye. Two sentences maximum."
        )
        return "Booking confirmed. Call ended."

    @function_tool()
    async def run_exit(self):
        """Prospect asked to be removed from call list."""
        await ExitTask(self.lead)
        self.call_result = 'suppressed'
        await self._end_call()
        return "Opted out. Suppression logged. Call ended."


# ============================================================
# LLM SELECTION
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# How long Ollama keeps qwen3:4b resident in memory after a request. Default
# (a few minutes) means the model unloads between calls during normal
# testing/campaign pacing, so the very failover this warm-up exists for
# would still hit a cold load. 30m comfortably spans a testing session.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


def _build_llm(provider: str):
    if provider == "groq":
        return groq.LLM(model="llama-3.3-70b-versatile", timeout=10.0)
    elif provider == "claude":
        return anthropic.LLM(model="claude-haiku-4-5")
    elif provider == "gpt":
        return openai_plugin.LLM(model="gpt-4o-mini")
    elif provider == "qwen":
        # qwen3:4b defaults to an internal "thinking" mode that generates a
        # long hidden reasoning chain before any tool call/content — this
        # measured ~45s to produce a single tool call locally, which reads
        # as total dead silence in a live call. enable_thinking=False (via
        # Ollama's OpenAI-compatible chat_template_kwargs pass-through)
        # drops that to ~4-5s. Confirmed empirically before relying on it.
        return openai_plugin.LLM(
            model="qwen3:4b",
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
        )
    elif provider == "mistral-local":
        # Stand-in for mistral-small until it's actually pulled in Ollama.
        return openai_plugin.LLM.with_ollama(model="gemma4:latest", base_url=OLLAMA_BASE_URL)
    elif provider == "gptoss":
        # Stand-in for gpt-oss:20b until it's actually pulled in Ollama.
        return openai_plugin.LLM.with_ollama(model="llama3.1:latest", base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print(f"=== Testing LLM: {provider.upper()} ===")

    primary = _build_llm(provider)

    # Groq is the one provider we've seen drop connections mid-session.
    # Rather than hand-rolled retry logic, wrap it in LiveKit's native
    # FallbackAdapter — it already retries with backoff and swaps to
    # the next LLM in the list on APIError/timeout (see
    # livekit.agents.llm.fallback_adapter.FallbackAdapter).
    failover_provider = os.getenv("LLM_FAILOVER_PROVIDER", "").lower()
    if provider == "groq" and failover_provider and failover_provider != provider:
        try:
            fallback = _build_llm(failover_provider)
        except Exception as e:
            print(f"WARNING: could not build LLM_FAILOVER_PROVIDER={failover_provider} ({e}); continuing without failover")
            return primary
        return FallbackAdapter(llm=[primary, fallback])

    return primary


# ============================================================
# TTS SELECTION
# ============================================================
#
# Neither Kokoro nor Chatterbox-Turbo has an official livekit-plugins
# package. Both are wired in via livekit-plugins-openai's TTS class
# pointed at a local OpenAI-compatible server (`/v1/audio/speech`),
# same pattern as get_llm() uses openai_plugin.LLM.with_ollama() for
# local models:
#   - Kokoro: run a Kokoro-FastAPI server (OpenAI-compatible) locally.
#   - Chatterbox-Turbo: the chatterbox-tts pip package is library-only
#     (no bundled server) — run it behind a community OpenAI-compatible
#     wrapper such as devnen/Chatterbox-TTS-Server.

KOKORO_BASE_URL     = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
CHATTERBOX_BASE_URL = os.getenv("CHATTERBOX_BASE_URL", "http://localhost:8880/v1")


def get_tts():
    provider = os.getenv("TTS_PROVIDER", "cartesia").lower()
    print(f"=== Testing TTS: {provider.upper()} ===")

    if provider == "cartesia":
        return cartesia.TTS(
            voice="a33f7a4c-100f-41cf-a1fd-5822e8fc253f",
            speed=1.0,
        )
    elif provider == "kokoro":
        return openai_plugin.TTS(
            model="kokoro",
            voice="af_bella",
            base_url=KOKORO_BASE_URL,
            api_key="not-needed",
        )
    elif provider == "chatterbox":
        return openai_plugin.TTS(
            model="chatterbox",
            voice="default",
            base_url=CHATTERBOX_BASE_URL,
            api_key="not-needed",
        )
    elif provider == "elevenlabs":
        return elevenlabs.TTS(
            voice_id="56bWURjYFHyYyVf490Dp",
            model="eleven_turbo_v2_5",
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {provider}")


# ============================================================
# STARTUP
# ============================================================

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

    lead = get_next_lead()

    if lead is None:
        print("No leads available.")
        return

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


if __name__ == "__main__":
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
