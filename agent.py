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
from livekit.agents.llm import FallbackAdapter, ChatMessage
from livekit.agents import inference
from livekit.plugins import groq, cartesia, deepgram, silero, anthropic, openai as openai_plugin, elevenlabs
from db import create_call, get_next_lead, update_lead_status, add_to_suppression
from call_transcript import CallTranscriptRecorder
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
     Example: "Rates run about 0.5 to 3 percent, but yeah — depends
     on the specifics."
   - Break up long sentences into two shorter ones with a pause
     between them rather than one long flowing sentence
   - Do NOT use exclamation points — they read as fake enthusiasm
   - A slight, natural stumble occasionally reads as more human than
     a perfectly smooth sentence: "We — we can definitely look into
     that for you."
"""

KNOWLEDGE_DEFER_RULE = """
ANSWER WHAT YOU KNOW, DEFER ONLY WHAT YOU DON'T:
Answer directly and confidently anything covered by your knowledge base or
this task's script — never defer something you actually know. If a question
(or part of a multi-part question) touches a specific fact, number, or stat
you don't actually have in your knowledge — answer whichever parts you DO
know directly, and for the specific unknown part only, say something like
"That exact number I'd have to check with our team, but I can tell you
[the part you do know]" — never invent a specific number or fact you don't
have, even for just one piece of a larger question.
"""

# _END_CALL CLOSING-LINE GUARD — prepended to every closing_instructions
# passed to _end_call(). Without this, the model sometimes treats the
# closing instruction as a suggestion and tacks on an off-script follow-up
# question instead of ending the call. This makes "no next turn, no
# question, say only this" explicit regardless of which LLM is active.
_FINAL_LINE_PREFIX = (
    "This is the last thing you say before the call ends — there is no "
    "next turn, so do not ask any question and do not offer anything "
    "further. Do not repeat your exact previous response word-for-word; "
    "vary the phrasing if it's similar to something already said. Say "
    "only the following, then stop: "
)


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


# ============================================================
# RATE-GUARDRAIL — TWO sanctioned ranges, told apart by context:
#   FEE     (what Porter charges, monthly, on outstanding invoices): 0.5-3%
#   ADVANCE (% of invoice value paid upfront):                       80-95%
# For either range, only the range itself — never a single confident
# number — is sanctioned dialogue. Context is decided by scanning a window
# of text around each matched number for keyword cues (_FEE_CONTEXT /
# _ADVANCE_CONTEXT below): "rate"/"fee"/"charge"/"percent...depends" mark a
# FEE mention, "advance"/"upfront"/"of your invoice" mark an ADVANCE
# mention. A number that can't be pinned to exactly one context (both cue
# sets hit, or neither) is treated conservatively as a violation — never
# guessed into whichever range happens to contain the value.
# Dollar figures (funding estimates) never match: the regex requires a
# trailing % or "percent", which those figures never have.
# ============================================================
_FEE_RANGE = (0.5, 3.0)
_ADVANCE_RANGE = (80.0, 95.0)

_NUM = r'\d+(?:\.\d+)?'
_RATE_RANGE = re.compile(rf'({_NUM})\s*(?:to|and|-|–|—)\s*({_NUM})\s*(?:%|percent\b)', re.IGNORECASE)
_RATE_SINGLE = re.compile(rf'({_NUM})\s*(?:%|percent\b)', re.IGNORECASE)
_RATE_HEDGE = re.compile(r'(?:around|about|approximately|roughly|somewhere\s+(?:around|between|near)|close\s+to)\s*$', re.IGNORECASE)

# Context cues, checked in a window of text around each matched number.
_FEE_CONTEXT = re.compile(r'\b(rate|rates|fee|fees|charge|charges|charging)\b', re.IGNORECASE)
_FEE_CONTEXT_DEPENDS = re.compile(r'percent\b.{0,40}?\bdepends\b', re.IGNORECASE | re.DOTALL)
_ADVANCE_CONTEXT = re.compile(r'\b(advance|advanced|advancing|upfront|up\s+front)\b|\bof\s+(?:your|the)\s+invoice\b', re.IGNORECASE)
_CONTEXT_WINDOW = 80  # chars of surrounding text scanned for context cues

RATE_FALLBACK_LINE = (
    "Rates typically run somewhere between 0.5 and 3 percent — "
    "I'll get you the exact number through one of our advisors."
)
ADVANCE_FALLBACK_LINE = (
    "We typically advance somewhere between 80 and 95 percent of the "
    "invoice value upfront — I'll get you the exact number through one "
    "of our advisors."
)


def _classify_rate_context(text: str, start: int, end: int) -> str:
    """Classify the number/range at text[start:end] as 'fee', 'advance', or
    'unclear', based on keyword cues in a window of surrounding text."""
    window = text[max(0, start - _CONTEXT_WINDOW): end + _CONTEXT_WINDOW]
    is_fee = bool(_FEE_CONTEXT.search(window) or _FEE_CONTEXT_DEPENDS.search(window))
    is_advance = bool(_ADVANCE_CONTEXT.search(window))
    if is_fee and not is_advance:
        return "fee"
    if is_advance and not is_fee:
        return "advance"
    return "unclear"


def find_rate_violation(text: str):
    """Return (description, fallback_line) for the first out-of-policy rate
    mention, or None. An unclassifiable context always counts as a
    violation (conservative default) and uses the FEE fallback line, since
    that's the older, more commonly triggered of the two rules."""
    range_spans = []
    for m in _RATE_RANGE.finditer(text):
        range_spans.append((m.start(), m.end()))
        low, high = float(m.group(1)), float(m.group(2))
        context = _classify_rate_context(text, m.start(), m.end())
        if context == "unclear":
            return f"unclassifiable rate range '{m.group(0).strip()}'", RATE_FALLBACK_LINE
        lo, hi = _FEE_RANGE if context == "fee" else _ADVANCE_RANGE
        fallback = RATE_FALLBACK_LINE if context == "fee" else ADVANCE_FALLBACK_LINE
        if not (lo <= low <= hi and lo <= high <= hi):
            return f"out-of-range {context} range '{m.group(0).strip()}'", fallback

    for m in _RATE_SINGLE.finditer(text):
        if any(start <= m.start() < end for start, end in range_spans):
            continue  # already covered as part of a range match above
        value = float(m.group(1))
        context = _classify_rate_context(text, m.start(), m.end())
        if context == "unclear":
            return f"unclassifiable rate '{m.group(0).strip()}'", RATE_FALLBACK_LINE
        lo, hi = _FEE_RANGE if context == "fee" else _ADVANCE_RANGE
        fallback = RATE_FALLBACK_LINE if context == "fee" else ADVANCE_FALLBACK_LINE
        if not (lo <= value <= hi):
            return f"out-of-range {context} rate '{m.group(0).strip()}'", fallback
        if not _RATE_HEDGE.search(text[:m.start()]):
            return f"unhedged firm {context} quote '{m.group(0).strip()}'", fallback

    return None


class SpeechSafetyMixin:
    """Mixed into every Agent/AgentTask so BOTH its spoken audio and the text
    logged as its conversation-history item pass through the same
    strip_leaked_meta_text + rate-guardrail filtering, regardless of which
    task is active.

    tts_node (audio) and transcription_node (chat-history/transcript text)
    each get their own independent tee'd copy of the same raw LLM text (see
    AgentActivity._pipeline_reply_task_impl's `tee = itertools.tee(text, 2)`
    in the livekit-agents SDK) — filtering only one of them left the other
    carrying the original, unfiltered text straight into
    conversation_item_added and the persisted transcript, so a detected
    rate violation was blocked from audio but still logged/persisted
    verbatim. Both nodes now run the identical filter over their own copy
    so the two always converge on the same final (possibly substituted)
    text.
    """

    @staticmethod
    async def _filtered_text(text):
        full_text = "".join([chunk async for chunk in text])
        cleaned = strip_leaked_meta_text(full_text)
        if not cleaned:
            return
        violation = find_rate_violation(cleaned)
        if violation:
            description, fallback_line = violation
            print(f"[RATE GUARDRAIL] blocked text ({description}): {cleaned!r}")
            cleaned = fallback_line
        yield cleaned

    async def tts_node(self, text, model_settings):
        return Agent.default.tts_node(self, self._filtered_text(text), model_settings)

    async def transcription_node(self, text, model_settings):
        return Agent.default.transcription_node(self, self._filtered_text(text), model_settings)

    def _finish_task(self, result):
        """Shared chokepoint for every task's completion path (task_complete
        tools, and any direct self.complete() call like OpenerTask.on_enter's
        hung_up path).

        self.complete() only resolves a future — it doesn't stop the SDK
        from generating a brand-new reply off this task's still-active
        instructions before the async handoff (session._update_activity,
        awaited in AgentTask.__await_impl's finally block) actually lands.
        That gap is where a stray extra turn can sneak in after the task has
        already "completed" (e.g. a redundant qualifying question spoken
        after task_complete('not_qualified') already fired). This must run
        synchronously, at complete()-time, not rely on the handoff timing.
        """
        if self.done():
            return
        activity = self._activity
        if activity is not None:
            # Sanctioned: the same call AgentTask.cancel() makes above to
            # kill whatever's currently in flight/queued on this activity.
            activity.interrupt(force=True)
            # ponytail: workaround, not a public SDK hook — _new_turns_blocked
            # has no supported synchronous setter (the real one, pause(), is
            # async and session-owned). Safe here because this task instance
            # is one-shot (AgentTask is not re-entrant, never resumed for new
            # turns), so permanently blocking is correct, not just temporary.
            # Upgrade path: swap for a real API if the SDK ever exposes one.
            activity._new_turns_blocked = True
        self.complete(result)


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
        company      = lead.get('company_name', 'your company')
        city         = lead.get('city', '')
        contact_name = lead.get('contact_name') or ''

        super().__init__(
            instructions=f"""
            You are Aiva, an AI sales assistant at Porter Capital.
            Calling: {company} in {city}
            Contact name (if known): {contact_name}

            {NATURAL_SPEECH_GUIDE}

            {KNOWLEDGE_DEFER_RULE}

            YOUR ONLY JOB: Open this call like a real human phone rep would
            — confirm you've reached the right company/person FIRST, and only
            disclose the AI/cold-call framing and ask for time AFTER that's
            confirmed. Never do both in the same breath.

            MANDATORY SEQUENCE — NO EXCEPTIONS: TURN 1 must be the literal
            first thing you say once the prospect responds to the hello
            check, every single time. Never skip straight to the
            RIGHT-PERSON CHECK or TURN 2 because you already know
            {company}'s name (or {contact_name}, if provided) from your own
            internal context above — that
            knowledge is for you only and never counts as confirmation.
            The ONLY thing that satisfies TURN 1 is the prospect themselves
            confirming it out loud, in this conversation.

            TURN 1 — COMPANY/CONTACT CONFIRMATION ONLY. Ask exactly this,
            nothing more (no disclosure, no Porter Capital mention yet):

            - IF {contact_name} IS PROVIDED (not empty/unknown):
              "Hey — is this {contact_name} over at {company}?"
            - IF {contact_name} IS NOT PROVIDED (empty/unknown):
              "Hey — I'm trying to reach {company}, is that who I've got?"

            Listen to their answer and branch:
            - REPEAT/CLARIFY REQUEST (they're asking you to repeat,
              rephrase, or clarify the question itself — e.g. "what?", "can
              you repeat that?", "sorry, say that again?", "huh?") -> this
              is NOT an answer to classify. Simply repeat or naturally
              rephrase the same question. Do NOT call task_complete. Do NOT
              treat this as confirmation, denial, unclear, or any other
              classification below — it's a request to hear the question
              again, nothing more.
            - CONFIRMED (yes, this is {company}/{contact_name}, or anything
              affirming it)
              -> move to the RIGHT-PERSON CHECK below, in your next reply.
              Do not call task_complete yet.
            - CONFUSED (they didn't catch it — "who's this?", "what
              company?", "sorry, what?") -> briefly repeat or rephrase the
              same confirmation question ONE time, naturally. Do not
              proceed to TURN 2 until they've actually confirmed or denied.
            - ANY NON-CONFIRMATION (this is TURN 1 ONLY — a plain "no,"
              "wrong number," "no one here by that name," or still unclear
              after one rephrase — every one of these means the same thing
              here: not yet a confirmed wrong number, just an unconfirmed
              one) -> do
              NOT call task_complete yet and do NOT apologize yet. Ask
              exactly ONE soft follow-up first, naturally, to rule out a
              mishearing or a subsidiary/different-location name mismatch:
              "Ah — are you with {company} at all, or is this a different
               business?"
              This is TURN 1B. Listen to their answer and branch:
              - REPEAT/CLARIFY REQUEST (they're asking you to repeat,
                rephrase, or clarify the question itself — e.g. "what?",
                "can you repeat that?", "sorry, say that again?", "huh?")
                -> this is NOT an answer to classify. Simply repeat or
                naturally rephrase the same question. Do NOT call
                task_complete. Do NOT treat this as confirmation, denial,
                unclear, or any other classification below — it's a request
                to hear the question again, nothing more.
              - They confirm they ARE with {company} after all -> this is
                NOT a wrong number. Move to the RIGHT-PERSON CHECK below, in
                your next reply, exactly as if they'd confirmed the first
                time. Do not call task_complete yet.
              - They confirm they are NOT with {company} (still no, wrong
                business, no one by that name) -> now it's a confirmed
                wrong number. Call the task_complete tool with result set
                to wrong_number, and do not speak anything yourself in this
                turn — no apology, no goodbye, no other words. The
                task_complete tool speaks the apology AND the warm sign-off
                itself, in a separate, guaranteed step, before the call ends.
              This is a dialing/data problem, not a sales decision — there
              is NO bad_timing/not_interested clarifying question at TURN 1,
              ever, no matter how the "no" is phrased.
              - UNCLEAR (their answer doesn't clearly fit CONFIRMED,
                CONFUSED, or NON-CONFIRMATION above — a mumbled answer,
                something off-topic, or anything you're not confident you
                understood correctly) -> do NOT guess which branch it is and
                do NOT call task_complete yet. Ask ONE brief clarifying
                question naturally, then follow whichever branch their
                answer now clearly fits. If it's STILL unclear after that
                one clarifying attempt, default to treating it as TURN 1B
                (ask if they're with {company} at all) rather than guessing
                confirmed or wrong_number, but never guess blindly.

              UNCLEAR ANSWER TO TURN 1B ITSELF (their answer to "are you
              with {company} at all, or is this a different business?"
              doesn't clearly confirm or deny) -> do NOT guess and do NOT
              call task_complete yet. Ask ONE brief clarifying question
              naturally, then follow whichever branch their answer now
              clearly fits. If it's STILL unclear after that one clarifying
              attempt, default to wrong_number — conservative, since you
              shouldn't force a conversation on someone who may not even be
              with the right company. Call the task_complete tool with
              result set to wrong_number, and do not speak anything yourself
              in this turn — no apology, no goodbye, no other words, same as
              the confirmed wrong_number case above — but never guess
              blindly before that one clarifying attempt.

            RIGHT-PERSON CHECK — once the company/contact is confirmed
            (either directly at TURN 1 or via TURN 1B), before disclosing
            anything about AI or asking for time, confirm you've got the
            right person. Ask exactly this:
            "I'm calling about working capital options for {company} — are
             you the right person for that, or is there someone else who
             handles it?"

            Listen to their answer and branch:
            - REPEAT/CLARIFY REQUEST (they're asking you to repeat,
              rephrase, or clarify the question itself — e.g. "what?", "can
              you repeat that?", "sorry, say that again?", "huh?") -> this
              is NOT an answer to classify. Simply repeat or naturally
              rephrase the same question. Do NOT call task_complete. Do NOT
              treat this as confirmation, denial, unclear, or any other
              classification below — it's a request to hear the question
              again, nothing more.
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
                name/contact info/department, verbatim or close to it). This
                referral info gets logged for a human advisor to follow up
                on directly.
              - They don't know who the right person is, or say no one else
                is available or relevant -> acknowledge warmly with a
                closing line yourself — e.g. "No worries at all, thanks for
                your time — have a good one!" — THEN call the task_complete
                tool with result set to not_interested.
              This is a routing dead end, not a sales objection — there is
              no bad_timing distinction here, only whether they gave you a
              referral or not.
            - UNCLEAR (doesn't clearly confirm or deny) -> do NOT call
              task_complete and do NOT guess which branch it is. Ask exactly
              ONE clarifying rephrase first, naturally: "Sorry, just to
              confirm — are you the right person to talk to about that, or
              should I ask for someone else?"
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
             I'm an AI, Aiva, calling for Porter Capital. Can I get 30
             seconds? I'll be crisp. No worries at all if now's not a good
             time."

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
            - The PROSPECT'S/LEAD'S company name ({company}) and contact
              name ({contact_name}, if provided) are only for the TURN 1
              confirmation question and the RIGHT-PERSON CHECK question —
              never say either again after that, they're internal context
              otherwise.
            - After they respond to TURN 2, call the task_complete tool —
              do not write it out as text. The tool call itself is a
              separate, silent action, never part of what you say out loud.
            - EVERY SINGLE ENDING of this task — wrong number, no referral
              given, referral given, right-person declined at TURN 2 — MUST
              include a warm sign-off (e.g. "have a good day," "take care,"
              "thanks so much") before the call ends. No silent or abrupt
              endings, ever, regardless of which branch is taken.
            - REPEAT/CLARIFY REQUEST (their response to TURN 2 is asking
              you to repeat, rephrase, or clarify what you just said — e.g.
              "what?", "can you repeat that?", "sorry, say that again?",
              "huh?") -> this is NOT an answer to classify. Simply repeat
              or naturally rephrase the same TURN 2 line. Do NOT call
              task_complete. Do NOT treat this as bad_timing,
              not_interested, or any other classification below — it's a
              request to hear it again, nothing more.
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
              - If it's a timing thing -> that's bad_timing.
              - If it's a genuine no -> that's not_interested.
            - If the prospect offers a SHORT WINDOW of time rather than
              declining (e.g., "I've got 10 seconds," "make it quick,"
              "you've got a minute," "go fast") — this is ACCEPTANCE, not
              bad_timing. Treat this as agreeing to listen, and move to the
              pitch — but deliver it in the MOST crisp, shortened form
              possible, respecting the time they offered. Do not call
              task_complete with bad_timing for this case.
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
            if attempt == 0:
                # HELLO-RACE GUARD — reaching this line already took two
                # sequential LLM round-trips (the start_call tool-call
                # decision, then this method being scheduled), on top of
                # room/audio connect. A prospect who answers instantly can
                # speak before any of that resolves, and STT commits it into
                # chat_ctx before Aiva has said a word. If that utterance is
                # left in context, the model conditions its "exactly Hello?"
                # generation on it and improvises a reply instead — which is
                # how a real prospect's own greeting can make Aiva skip
                # straight past the scripted opener. Anything already in
                # chat_ctx at this point predates Aiva's first spoken word,
                # so it isn't a reply to anything she's said and must not
                # influence what she says first — strip it before asking
                # for the hello generation.
                stale_user_turns = [
                    item for item in self.chat_ctx.items
                    if isinstance(item, ChatMessage) and item.role == "user"
                ]
                if stale_user_turns:
                    print(
                        f"[OpenerTask] discarding {len(stale_user_turns)} "
                        "user turn(s) that arrived before Aiva's first "
                        "spoken word"
                    )
                    trimmed_ctx = self.chat_ctx.copy()
                    trimmed_ctx.items = [
                        item for item in trimmed_ctx.items
                        if not (isinstance(item, ChatMessage) and item.role == "user")
                    ]
                    await self.update_chat_ctx(trimmed_ctx)
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
                        self._finish_task('hung_up')
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
            self._finish_task({'result': result, 'referral_details': referral_details})
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
                instructions='Apologize briefly for the mix-up and add a warm '
                'sign-off — e.g. "Oh — sorry about that, my mistake. Have a '
                'good day." Keep it short, then stop.',
                allow_interruptions=False,
            )
        if not self.done():
            self._finish_task(result)


# ============================================================
# TASK 2 — PITCH
# ============================================================

class PitchTask(SpeechSafetyMixin, AgentTask):

    def __init__(self, lead):
        super().__init__(
            instructions=f"""
            You are Aiva. The prospect just agreed to listen (said yes to
            the ice breaker).

            NOTE: if they agreed by offering a SHORT WINDOW of time rather
            than a plain yes (e.g. "I've got 10 seconds," "make it quick,"
            "you've got a minute," "go fast"), deliver the pitch even more
            tightly than usual — prioritize the core message (who you help,
            the one differentiator, the closing check-in question) over
            full script fidelity if time is explicitly limited.

            You are mid-call. The prospect already heard the ice breaker and
            AI disclosure from earlier in this conversation. Never repeat
            the opening greeting, never re-introduce yourself, never
            re-disclose being an AI unless directly asked again. Jump
            straight into your own job.

            {NATURAL_SPEECH_GUIDE}

            {KNOWLEDGE_DEFER_RULE}

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
            "So — Porter Capital here. Basically, we help businesses get
             working capital — cash against their unpaid invoices, instead
             of waiting to get paid. We've actually been doing this for
             decades — funded billions to businesses out there. Worth a
             quick chat?"

            The closing line is always some version of "worth a quick
            chat?" — a soft check-in, never "what's your current process
            for X" or any other fact-finding question. Ending the pitch on
            a qualifying-style question is a rule violation, not an
            acceptable variation.

            If asked how fast funding happens: "Once you're set up with us,
            we move fast — usually under 48 hours from submitting an
            invoice." This is the only funding-speed detail you may
            give — never imply this applies to a brand-new prospect's
            first-ever funding from this call.

            If asked how much of the invoice gets advanced upfront: "We
            typically advance somewhere between 80 and 95 percent of the
            invoice value upfront — the exact number depends on your
            customers and how the deal is structured." Only say this if
            they actually ask — never volunteer it as part of the pitch.

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
            - NEVER quote rates or dollar amounts EXCEPT the advance-rate
              answer above, and only if asked directly — that's the one
              sanctioned exception, never volunteered.

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

            If their response is unclear or doesn't clearly fit interested
            or objection — a mumbled answer, something off-topic, or
            anything you're not confident you understood correctly — do NOT
            guess which branch it is and do NOT call task_complete yet. Ask
            ONE brief clarifying question naturally, e.g. "Sorry, was that a
            yes, or were you not so sure?", then follow whichever branch
            their answer now clearly fits. If it's STILL unclear after that
            one clarifying attempt, default to treating it as an objection —
            call task_complete with result set to objection and
            objection_text noting the response was unclear, rather than
            forcing interested; it's safer to route to a human-guided
            objection-handling conversation than assume interest that
            wasn't really expressed, but never guess blindly.
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
            self._finish_task({"result": result, "objection_text": objection_text})
        else:
            self._finish_task({"result": result, "objection_text": ""})


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

            {KNOWLEDGE_DEFER_RULE}

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

            If their answer to either the business-type question or the
            factoring question is unclear or doesn't clearly fit what
            you're asking — a mumbled answer, something off-topic, or
            anything you're not confident you understood correctly — do NOT
            guess and do NOT call task_complete yet. Ask ONE brief
            clarifying question naturally, then follow whichever branch
            their answer now clearly fits. If it's STILL unclear after that
            one clarifying attempt, stay in this task and keep asking
            rather than guessing or completing the task — there's no
            urgency to end the call here, so never guess blindly just to
            move forward.

            ALREADY FACTORING WITH SOMEONE ELSE:
            This is NOT an automatic disqualifier — do not treat it as a
            dead end. A prospect who already has a factor may still be
            unhappy with them or open to switching, and there's a dedicated
            objection script for exactly this. If they say they already
            factor invoices with another company, call the task_complete
            tool with result set to objection and objection_text set to
            what they said — this hands off to the objection-handling flow
            (the "we already have a factor" script), not an instant
            not_qualified.
            Reserve not_qualified for when business type or factoring
            status genuinely rules them out for reasons OTHER than already
            having a factor (e.g. wrong type of business, or a clear
            negative signal unrelated to having an existing factor).

            OFFERING A ROUGH FUNDING ESTIMATE (only after business type and
            factoring status are established, and only for prospects who
            are NOT being routed to objection handling):

            Offer to give them a rough estimate — do not ask for their
            numbers upfront. Example:
            "I can actually give you a rough estimate of what we could get
            you, if that'd help — want me to?"

            If they say NO or decline:
            Skip the estimate entirely. Move directly to offering to connect
            them with a human advisor, same as usual.

            If they say YES:
            Ask for either their open accounts receivable (AR) balance or
            their annual revenue — whichever feels more natural, don't ask
            for both back to back.

            If the number they give you is unclear or garbled — a mumbled
            figure, or anything you're not confident you understood
            correctly — do NOT guess the number. Ask ONE brief clarifying
            question naturally, e.g. "Sorry, could you say that number
            again?", then use whichever number they now clearly give you.
            If it's STILL unclear after that one clarifying attempt, skip
            the estimate entirely and move to offering the advisor
            connection — same as if they'd said no to the estimate offer —
            rather than guessing at a number.

            Once you have ONE of these numbers, calculate a rough estimate:
            - If given an AR balance: estimate = that number x 90%
            - If given annual revenue: estimate = that number x 10%
            - If given both: use either one, your choice

            State this as a rough, non-binding estimate — always immediately
            followed by an offer to connect them with a human advisor for
            exact numbers. Example phrasing:
            "Based on that, we could likely get you up to around [estimate]
            — but the exact number really depends on the details, so let's
            get you connected with one of our advisors who can nail that
            down for you. Sound good?"

            NEVER present this estimate as a guaranteed or final number —
            it is always a rough approximation, and the human advisor
            always provides the real figure.

            If they agree to connect with an advisor (whether or not they
            took the estimate), move to booking — get their best day/time
            for a callback, same as the existing "open to exploring" flow.

            OFF-SCRIPT QUESTIONS (process, eligibility, rates):
            An engaged prospect asking questions about the process,
            eligibility, or rates is NOT a disqualifying signal — it's
            often a sign of genuine interest. Answer briefly, defer
            specifics to the human advisor, then return to qualifying or
            move toward booking if they seem ready. Do NOT call
            task_complete just because the conversation went off-script —
            only call it once you've actually determined qualified,
            not_qualified, or objection (see ALREADY FACTORING above) based
            on business type and factoring status (or a clear negative
            signal from the prospect).

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
            - NEVER quote rates, percentages, or dollar amounts EXCEPT the
              rough funding estimate above, which is the one sanctioned
              exception, and must always be framed as non-binding.
            - If asked about rates specifically (not the funding estimate),
              say only:
              "Honestly, it depends on a few things — how much you're
               invoicing, your customers, stuff like that. Rates typically
               run somewhere between 0.5 and 3 percent... but we'd build you
               an actual number once we know more."
            - If asked how much of the invoice gets advanced upfront:
              "We typically advance somewhere between 80 and 95 percent of
               the invoice value upfront — the exact number depends on your
               customers and how the deal is structured."
            - If asked how fast funding happens: "Once you're set up with
              us, we move fast — usually under 48 hours from submitting an
              invoice."
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
    async def task_complete(self, result: str, objection_text: str = ""):
        """result: qualified / not_qualified / objection.
        objection_text: what they said, only when result is objection
        (e.g. they already factor invoices with someone else)."""
        if result == "objection":
            self._finish_task({"result": result, "objection_text": objection_text})
        else:
            self._finish_task({"result": result, "objection_text": ""})


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

            {KNOWLEDGE_DEFER_RULE}

            Follow this formula STRICTLY: Acknowledge -> Ask a question -> Continue.
            Never defend, never argue, never over-explain. One acknowledgment,
            one question, then stop.

            Examples:
            "We already have a factor" ->
            "That makes sense... most people do. Worth a quick look to see
             if we could actually do better for you?"

            "Not interested" ->
            "Totally fair. Can I ask real quick — is it bad timing, or just
             not something you need right now?"
            Then listen to which one it is — this single answer decides the
            result you pass:
            - If it's a timing thing ("busy right now", "call me later",
              "not a good time", "maybe down the road") -> that's bad_timing.
              Once you know it's a timing thing, ask: "No worries — when
              would be a better time for us to check back in?" and note
              whatever they say (e.g. "next month", "in two weeks") so it
              can be passed along.
            - If it's a genuine no ("just don't need it", "we're all set",
              "not for us") -> that's not_interested.

            If their answer to "is it bad timing, or just not something you
            need right now?" is unclear or doesn't clearly fit either
            option — do NOT guess and do NOT call task_complete yet. Ask
            ONE brief clarifying rephrase naturally, then follow whichever
            result their answer now clearly fits. If it's STILL unclear
            after that one clarifying attempt, default to not_interested —
            conservative, since you shouldn't assume a callback was wanted
            if it wasn't clearly stated — but still end warmly, same as the
            existing not_interested pattern, rather than guessing blindly.

            "What are your rates" ->
            "Honestly, it depends on a few things — how much you're
             invoicing, your customers, stuff like that. Rates typically
             run somewhere between 0.5 and 3 percent... but we'd build you
             an actual number once we know more."

            "How much of my invoice do you advance" ->
            "We typically advance somewhere between 80 and 95 percent of
             the invoice value upfront — the exact number depends on your
             customers and how the deal is structured."

            "How fast could I get funded" ->
            "Once you're set up with us, we move fast — usually under 48
             hours from submitting an invoice."

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
            - NEVER quote a specific rate or percentage beyond the approved
              0.5-to-3-percent range shown in the script above. Never give a
              number more precise than that range. The 80-to-95-percent
              advance range above is a separate, also-approved number —
              never confuse the two or blend them into one figure.
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
        self._finish_task(result)


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

            {KNOWLEDGE_DEFER_RULE}

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

            If asked about rates: "Honestly, it depends on a few things —
            how much you're invoicing, your customers, stuff like that.
            Rates typically run somewhere between 0.5 and 3 percent... but
            we'd build you an actual number once we know more."

            If asked how much of the invoice gets advanced upfront: "We
            typically advance somewhere between 80 and 95 percent of the
            invoice value upfront — the exact number depends on your
            customers and how the deal is structured."

            If asked how fast funding happens: "Once you're set up with us,
            we move fast — usually under 48 hours from submitting an
            invoice."

            If anything they say is unclear or doesn't obviously fit what
            you're asking for (a garbled number, an ambiguous day/time, or
            anything you're not confident you understood correctly) — do
            NOT guess or make something up. Ask ONE brief clarifying
            question naturally, e.g. "Sorry, could you say that number
            again?" or "Just to confirm, did you mean [X]?" Never proceed
            with uncertain details.

            Confirm back naturally, then call the task_complete tool — do
            not write it out as text. Speak only your natural confirmation;
            the tool call itself is a separate, silent action, never part
            of what you say out loud.

            RULES:
            - Never say the PROSPECT'S/LEAD'S company name back to them — it's
              internal context only, never spoken aloud. (Porter Capital,
              Aiva's own employer, is separate and fine to say if relevant.)
            - Never quote rates beyond the approved 0.5-to-3-percent range,
              and never promise approval. The 80-to-95-percent advance
              range above is a separate, also-approved number — never
              confuse the two or blend them into one figure.
            - Keep it warm and brief — they already said yes
            """.replace("{NATURAL_SPEECH_GUIDE}", NATURAL_SPEECH_GUIDE)
            .replace("{KNOWLEDGE_DEFER_RULE}", KNOWLEDGE_DEFER_RULE)
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="React warmly to their agreement. Ask for callback preference naturally, using the soft closing line about getting them set up with an advisor."
        )

    @function_tool()
    async def task_complete(self, callback_details: str):
        """Call when callback details are confirmed."""
        self._finish_task(f"booked:{callback_details}")


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

            {KNOWLEDGE_DEFER_RULE}

            Answer like this — make it your own version:
            "I am, yeah... happy to keep going, or I can grab you one of
             our advisors if you'd rather talk to an actual person."

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
            - If their reaction is unclear or doesn't clearly fit accepted,
              wants_human, or wants_to_end — a mumbled answer, something
              off-topic, or anything you're not confident you understood
              correctly — do NOT guess and do NOT call task_complete yet.
              Ask ONE brief clarifying check-in naturally, e.g. "Are you
              good to keep chatting?", then follow whichever branch their
              answer now clearly fits. If it's STILL unclear after that one
              clarifying attempt, default to accepted — assume they're fine
              continuing — rather than guessing blindly.
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
        self._finish_task(prospect_reaction)


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

            {KNOWLEDGE_DEFER_RULE}

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
        if os.getenv("TEST_MODE", "false").lower() == "true":
            print(
                "TEST MODE — suppression_list NOT updated. Would have added: "
                f"company_name={self.lead.get('company_name', '')!r} "
                f"website_domain={self.lead.get('website_domain', '')!r} reason='opted_out'"
            )
        else:
            add_to_suppression(
                company_name   = self.lead.get('company_name', ''),
                website_domain = self.lead.get('website_domain', ''),
                reason         = 'opted_out'
            )
        self._finish_task('suppressed')


# ============================================================
# SUPERVISOR
# ============================================================

class PorterSupervisor(SpeechSafetyMixin, Agent):

    def __init__(self, lead, ctx: JobContext):
        self.lead        = lead
        self.ctx         = ctx
        self.call_result = 'no_answer'

        # Persisted alongside call_result at call-end (see when_call_starts).
        # objection_text holds only the MOST RECENT objection — a call can
        # pass through run_objection more than once, but the last one is the
        # most relevant context for a human advisor reviewing the lead, and
        # overwriting is simpler than concatenating a history.
        self.referral_details = None
        self.callback_details = None
        self.objection_text   = None

        company  = lead.get('company_name', 'the company')
        city     = lead.get('city', '')
        state    = lead.get('state', '')
        industry = lead.get('industry', 'staffing')
        tier     = lead.get('tier', 'warm')

        super().__init__(
            instructions=f"""
            Porter Capital sales call. Agent: Aiva (AI).
            Lead: {company}, {city} {state}, {industry}, Tier: {tier}

            ABSOLUTE RULE: NEVER quote rates or numbers except QualifierTask's
            explicitly sanctioned rough, non-binding funding estimate.
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
                instructions=_FINAL_LINE_PREFIX + closing_instructions,
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
            # Not a sales decision — a routing dead end. call_result falls
            # back to 'contacted' (real contact was made, just not with a
            # decision maker) rather than inventing a new status literal
            # db.py's update_lead_status() doesn't accept. referral_details
            # itself is persisted to its own column at call-end (see
            # when_call_starts) so a human advisor can actually act on it.
            referral_details = result.get('referral_details', '')
            print(f"[Gatekeeper referral] {self.lead.get('company_name', 'lead')}: {referral_details}")
            self.referral_details = referral_details
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
        self.objection_text = objection
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
        if result['result'] == 'qualified':
            return "Qualified. Move toward booking."
        if result['result'] == 'objection':
            # e.g. "we already have a factor" — a switchable objection, not
            # an automatic disqualifier. Route to the same objection-handling
            # flow run_pitch uses, rather than hanging up.
            return await self.run_objection(result['objection_text'])
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
        self.callback_details = callback_details
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
    elif provider == "grok":
        grok_model = os.getenv("GROK_MODEL", "xai/grok-4-1-fast-non-reasoning")
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            # Talk to xAI directly, bypassing LiveKit Inference.
            return openai_plugin.LLM(
                model=grok_model.removeprefix("xai/"),
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
                timeout=10.0,
            )
        # Default: LiveKit Inference gateway, authenticated via LIVEKIT_API_KEY/SECRET.
        return inference.LLM(model=grok_model)
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

    call_id = await asyncio.to_thread(
        create_call,
        lead["lead_candidate_id"],
        "v1",
        os.getenv("LLM_PROVIDER", "groq").lower(),
        os.getenv("TTS_PROVIDER", "cartesia").lower(),
        ctx.job.room.sid or ctx.room.sid or ctx.room.name,
    )

    pipeline = AgentSession(
        stt = deepgram.STT(model="nova-2"),
        llm = get_llm(),
        tts = get_tts(),
        vad = silero.VAD.load(),
    )

    supervisor = PorterSupervisor(lead, ctx)

    task_names = {
        "OpenerTask": "opener",
        "PitchTask": "pitch",
        "QualifierTask": "qualifier",
        "ObjectionTask": "objection",
        "BookingTask": "booking",
        "DisclosureTask": "disclosure",
        "ExitTask": "exit",
        "PorterSupervisor": "supervisor",
    }

    def _active_task():
        current_agent = pipeline.current_agent
        return task_names.get(
            type(current_agent).__name__,
            type(current_agent).__name__,
        )

    transcript = CallTranscriptRecorder(call_id, _active_task)
    pipeline.on(
        "conversation_item_added",
        transcript.on_conversation_item_added,
    )

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

    try:
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
    finally:
        await transcript.close(supervisor.call_result)

    if os.getenv("TEST_MODE", "false").lower() == "true":
        print(
            f"TEST MODE — lead status NOT updated. Would have set: {supervisor.call_result} | "
            f"referral_details={supervisor.referral_details!r} | "
            f"callback_details={supervisor.callback_details!r} | "
            f"objection_text={supervisor.objection_text!r}"
        )
    else:
        update_lead_status(
            lead_candidate_id = lead['lead_candidate_id'],
            new_status        = supervisor.call_result,
            referral_details  = supervisor.referral_details,
            callback_details  = supervisor.callback_details,
            objection_text    = supervisor.objection_text,
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
