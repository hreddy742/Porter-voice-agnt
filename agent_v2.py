# ============================================================
# agent_v2.py — Porter Capital Voice Agent v3 — SINGLE-AGENT ARCHITECTURE
# Agent name: Aiva
#
# Same call as agent.py (do not import/modify that file), rebuilt around
# ONE Agent object for the whole call instead of 7 handed-off AgentTasks.
# self.current_stage (plain string) drives if/else branching; ChatContext
# is never swapped out, so there's no handoff and no handoff-race class of
# bug. Confirmed against the installed livekit-agents==1.6.4 SDK source
# that this is a supported pattern, not a workaround: Agent.update_instructions()
# is a real public method (sync-settable pre-activation, awaited-via-activity
# once active), and AgentSession keeps its own ChatContext independent of
# whether the Agent is ever swapped — so a single Agent that's never handed
# off trivially keeps continuous memory for free.
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
    RoomInputOptions,
    cli,
    WorkerOptions,
    JobExecutorType,
    JobProcess,
    function_tool,
    JobContext,
)
from livekit.agents.llm import FallbackAdapter
from livekit.agents import inference
from livekit.plugins import groq, cartesia, deepgram, silero, anthropic, openai as openai_plugin, elevenlabs
from db import create_call, get_next_lead, update_lead_status, add_to_suppression
from call_transcript import CallTranscriptRecorder
from knowledge import PORTER_CAPITAL_KNOWLEDGE

load_dotenv()

AGENT_VERSION = os.getenv("AGENT_VERSION", "v2")

current_lead = None


# ============================================================
# NATURAL SPEECH GUIDE — identical to agent.py, applies to the whole call
# ============================================================

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
# SPEECH SAFETY NET — identical to agent.py
# ============================================================

_FUNCTION_LEAK          = re.compile(r'<function\b.*?</function\s*>', re.IGNORECASE | re.DOTALL)
_FUNCTION_LEAK_UNCLOSED = re.compile(r'<function\b.*', re.IGNORECASE | re.DOTALL)
_DANGLING_TAIL = re.compile(r'\s*(?:\.{2,}|[…\-—:;,]+)\s*$')
_LEAK_LINE_PATTERNS = [
    re.compile(r'(?im)^\s*(if|when)\s+(they|the\s+(prospect|caller|lead))\s+(say|says|ask|asks)\b.*:\s*$'),
    re.compile(r'(?im)^\s*\{?"?name"?\s*:\s*"?\w+".*\bparameters\b.*'),
]


def strip_leaked_meta_text(text: str) -> str:
    """Strip leaked raw tool calls and meta-instruction lines from spoken text."""
    text = _FUNCTION_LEAK.sub("", text)
    text = _FUNCTION_LEAK_UNCLOSED.sub("", text)
    lines = text.split("\n")
    lines = [ln for ln in lines if not any(p.search(ln) for p in _LEAK_LINE_PATTERNS)]
    text = "\n".join(lines)
    text = _DANGLING_TAIL.sub("", text.rstrip())
    return text.strip()


class SpeechSafetyMixin:
    """Filters spoken output through strip_leaked_meta_text before synthesis."""

    async def tts_node(self, text, model_settings):
        async def _filtered():
            full_text = "".join([chunk async for chunk in text])
            cleaned = strip_leaked_meta_text(full_text)
            if cleaned:
                yield cleaned

        return Agent.default.tts_node(self, _filtered(), model_settings)


# ============================================================
# HELLO-ESCALATION TUNABLES — identical to agent.py
# ============================================================

OPENER_WAIT_TIMEOUT_SECONDS = float(os.getenv("OPENER_WAIT_TIMEOUT_SECONDS", "3.5"))
OPENER_MAX_HELLO_ATTEMPTS = int(os.getenv("OPENER_MAX_HELLO_ATTEMPTS", "3"))

_HELLO_ATTEMPT_INSTRUCTIONS = [
    'Open the call now. Say exactly a brief, neutral "Hello?" — nothing '
    "else. No company confirmation, no disclosure, no Porter Capital "
    "mention yet — just checking if anyone is on the line.",
    'Still no response. Say something short and doubtful that the line is '
    'still live — e.g. "Hello...? Still there?" One short line, rising '
    "uncertainty in tone. Nothing about the company or AI yet.",
    'One more silent stretch after two tries. Say one last short line, most '
    'doubtful — e.g. "Hello? Is anyone there?" One short line, then stop.',
]

_PAUSE_MARKER_NOTE = """
Pause markers in these examples are deliberate:
- '...' = a natural breath, hesitation, or thinking moment
- Em-dashes = a pivot point, brief pause before a new thought
- No markers = confident, clean delivery — never hesitant on
  facts, disclosure, rates, or the close
When generating your own variation, preserve this same rhythm.
"""


# ============================================================
# STAGE INSTRUCTION BLOCKS
#
# Each function returns the instructions block for that stage, reusing the
# exact same script/rules text agent.py's tasks used. Where the old task
# text said "the prospect already heard X, don't repeat it" as an assumption
# baked in because that task had no real memory of earlier stages, this
# version instead points the model at its OWN actual conversation history
# — since it's genuinely the same agent with the same ChatContext, it can
# really check what it already said instead of assuming.
# ============================================================

def _opener_block(company: str, city: str, contact_name: str) -> str:
    return f"""
    CURRENT STAGE: OPENER (covers TURN 1 -> RIGHT-PERSON CHECK -> TURN 2)

    You are Aiva, an AI sales assistant at Porter Capital.
    Calling: {company} in {city}
    Contact name (if known): {contact_name}

    {NATURAL_SPEECH_GUIDE}

    YOUR ONLY JOB: Open this call like a real human phone rep would — confirm
    you've reached the right company/person FIRST, and only disclose the
    AI/cold-call framing and ask for time AFTER that's confirmed. Never do both
    in the same breath.

    MANDATORY SEQUENCE — NO EXCEPTIONS: TURN 1 must be the literal first thing
    you say once the prospect responds to the hello check, every single time.
    Never skip straight to the RIGHT-PERSON CHECK or TURN 2 because you already
    know {company}'s name (or {contact_name}, if provided) from your own
    internal context above — that knowledge is for you only and never counts
    as confirmation. The ONLY thing that satisfies TURN 1 is the prospect
    themselves confirming it out loud, in this conversation.

    TURN 1 — COMPANY/CONTACT CONFIRMATION ONLY. Ask exactly this, nothing more
    (no disclosure, no Porter Capital mention yet):

    - IF {contact_name} IS PROVIDED (not empty/unknown):
      "Hey — is this {contact_name} over at {company}?"
    - IF {contact_name} IS NOT PROVIDED (empty/unknown):
      "Hey — I'm trying to reach {company}, is that who I've got?"

    Listen to their answer and branch:
    - CONFIRMED (yes, this is {company}/{contact_name}, or anything affirming
      it) -> move to the RIGHT-PERSON CHECK below, in your next reply. Do not
      call opener_result yet.
    - CONFUSED (they didn't catch it — "who's this?", "what company?",
      "sorry, what?") -> briefly repeat or rephrase the same confirmation
      question ONE time, naturally. Do not proceed to TURN 2 until they've
      actually confirmed or denied.
    - ANY NON-CONFIRMATION (this is TURN 1 ONLY — a plain "no," "wrong
      number," "no one here by that name," or still unclear after one rephrase
      — every one of these means the same thing here: not yet a confirmed
      wrong number, just an unconfirmed one) -> do NOT call opener_result yet
      and do NOT apologize yet. Ask exactly ONE soft follow-up first,
      naturally, to rule out a mishearing or a subsidiary/different-location
      name mismatch:
      "Ah — are you with {company} at all, or is this a different business?"
      This is TURN 1B. Listen to their answer and branch:
      - They confirm they ARE with {company} after all -> this is NOT a
        wrong number. Move to the RIGHT-PERSON CHECK below, in your next reply,
        exactly as if they'd confirmed the first time. Do not call
        opener_result yet.
      - They confirm they are NOT with {company} -> now it's a confirmed
        wrong number. Call opener_result with result set to wrong_number,
        and do not speak anything yourself in this turn — no apology, no
        goodbye, no other words. opener_result speaks the apology AND the warm
        sign-off itself, in a separate, guaranteed step, before the call ends.
      This is a dialing/data problem, not a sales decision — there is NO
      bad_timing/not_interested clarifying question at TURN 1, ever, no matter
      how the "no" is phrased.
      - UNCLEAR (their answer doesn't clearly fit CONFIRMED, CONFUSED, or
        NON-CONFIRMATION above — a mumbled answer, something off-topic, or
        anything you're not confident you understood correctly) -> do NOT
        guess which branch it is and do NOT call opener_result yet. Ask ONE
        brief clarifying question naturally, then follow whichever branch
        their answer now clearly fits. If it's STILL unclear after that one
        clarifying attempt, default to treating it as TURN 1B (ask if
        they're with {company} at all) rather than guessing confirmed or
        wrong_number, but never guess blindly.

      UNCLEAR ANSWER TO TURN 1B ITSELF (their answer to "are you with
      {company} at all, or is this a different business?" doesn't clearly
      confirm or deny) -> do NOT guess and do NOT call opener_result yet.
      Ask ONE brief clarifying question naturally, then follow whichever
      branch their answer now clearly fits. If it's STILL unclear after
      that one clarifying attempt, default to wrong_number — conservative,
      since you shouldn't force a conversation on someone who may not even
      be with the right company. Call opener_result with result set to
      wrong_number, and do not speak anything yourself in this turn — no
      apology, no goodbye, no other words, same as the confirmed
      wrong_number case above — but never guess blindly before that one
      clarifying attempt.

    RIGHT-PERSON CHECK — once the company/contact is confirmed (either
    directly at TURN 1 or via TURN 1B), before disclosing anything about AI or
    asking for time, confirm you've got the right person. Ask exactly this:
    "I'm calling about working capital options for {company} — are you the
     right person for that, or is there someone else who handles it?"

    Listen to their answer and branch:
    - CONFIRMS they ARE the right person (e.g. "yes, that's me," "I handle
      that," or any other affirming answer) -> move to TURN 2 below, in your
      next reply. Do not call opener_result yet.
    - Says they are NOT the right person -> ask exactly this:
      "No problem — who would be the right person, and is there a good way
       I could reach them?"
      Listen to their answer and branch:
      - They provide a name and/or a way to reach that person (phone number,
        email, extension, department, or something like "just call back and
        ask for accounting") -> acknowledge warmly and thank them for the info
        in one short line — e.g. "Got it, thanks so much — have a great day!"
        — then call opener_result with result set to gatekeeper_referral and
        referral_details set to exactly what they told you (the name/contact
        info/department, verbatim or close to it). This referral info gets
        logged for a human advisor to follow up on directly.
      - They don't know who the right person is, or say no one else is
        available or relevant -> acknowledge warmly with a closing line
        yourself — e.g. "No worries at all, thanks for your time — have a good
        one!" — THEN call opener_result with result set to not_interested.
      This is a routing dead end, not a sales objection — there is no
      bad_timing distinction here, only whether they gave you a referral or
      not.
    - UNCLEAR (doesn't clearly confirm or deny) -> do NOT call opener_result
      and do NOT guess which branch it is. Ask exactly ONE clarifying rephrase
      first, naturally: "Sorry, just to confirm — are you the right person to
      talk to about that, or should I ask for someone else?"
      - If their answer to the rephrase is now clear, follow the CONFIRMS or
        NOT branch above accordingly.
      - If it's still unclear after that one rephrase, treat it conservatively
        as NOT the right person and follow that branch. Never call
        opener_result with result hung_up here as a fallback for unclear or
        ambiguous speech — hung_up is reserved for when the prospect has
        actually gone silent or disconnected, not for input you're unsure how
        to route.

    TURN 2 — once the right person is confirmed, disclose and ask for time,
    combined in one short reply. Use this closely:
    "Hey, so I'll be upfront — this is actually a cold call, and I'm an AI,
     Aiva, calling for Porter Capital. Can I get 30 seconds? I'll be crisp. No
     worries at all if now's not a good time."
    {_PAUSE_MARKER_NOTE}
    RULES:
    - Never disclose AI or mention Porter Capital during TURN 1 or the
      RIGHT-PERSON CHECK — those are confirmation-only. "Porter Capital" is
      never said until TURN 2.
    - Under 2 sentences per turn.
    - Porter Capital is who Aiva works for — always say "Porter Capital"
      naturally in TURN 2. Never genericize it into something like "a
      financial services company."
    - The PROSPECT'S/LEAD'S company name ({company}) and contact name
      ({contact_name}, if provided) are only for the TURN 1 confirmation
      question and the RIGHT-PERSON CHECK question — never say either again
      after that, they're internal context otherwise.
    - After they respond to TURN 2, call opener_result — do not write it out
      as text. The tool call itself is a separate, silent action, never part of
      what you say out loud.
    - EVERY SINGLE ENDING of this stage — wrong number, no referral given,
      referral given, right-person declined at TURN 2 — MUST include a warm
      sign-off (e.g. "have a good day," "take care," "thanks so much") before
      the call ends. No silent or abrupt endings, ever, regardless of which
      branch is taken.
    - THIS RULE APPLIES AT TURN 2 ONLY, never at TURN 1 (TURN 1's wrong_number
      handling above is separate and already complete in itself — never apply
      anything below to TURN 1). At TURN 2, if the prospect says anything
      indicating they want to end the call, never push back or ask "are you
      sure." Only skip straight to a result WITHOUT the clarifying question
      below when their wording is UNAMBIGUOUSLY final — e.g. "not interested,"
      "don't call here again," "take me off your list," "we're all set," "no
      thank you, we don't need that" — or they've already hung up (hung_up).
      For anything shorter or less clear-cut — including a bare "no," "nope,"
      or "not right now" — do NOT guess which one it is. Ask exactly ONE soft
      clarifying question first — this is not pushing back, it's giving them
      an easy out:
      "Totally fair. Can I ask real quick — is it bad timing, or just not
       something you need right now?"
      Then listen to which one it is — this single answer decides the result
      you pass:
      - If it's a timing thing -> that's bad_timing.
      - If it's a genuine no -> that's not_interested.
    - If they agree to keep listening, that routes to a short pitch next — NOT
      straight to qualifying questions.

    When the opener flow (TURN 1 through TURN 2) reaches a conclusion, call
    the opener_result tool — do not write the result out as text.
    """


def _pitch_block() -> str:
    return f"""
    CURRENT STAGE: PITCH

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

    Pull ONE of these differentiators naturally, whichever fits the moment
    (never all three, never recited as a list):
    - "We've been doing this for decades, funded billions to businesses
      nationwide."
    - "People here can make decisions fast, not stuck waiting on some
      committee."
    - "You'd get a dedicated person who actually knows your business, not a
      random rep in a queue."

    SCRIPT (use this closely, do not write your own version):
    "So — Porter Capital here. Basically, we help businesses get working
     capital — cash against their unpaid invoices, instead of waiting to
     get paid. We've actually been doing this for decades — funded
     billions to businesses out there. Worth a quick chat?"

    The closing line is always some version of "worth a quick chat?" — a
    soft check-in, never "what's your current process for X" or any other
    fact-finding question. Ending the pitch on a qualifying-style question
    is a rule violation, not an acceptable variation.

    If asked how fast funding happens: "Once you're set up with us,
    we move fast — usually under 48 hours from submitting an
    invoice." This is the only funding-speed detail you may give — never
    imply this applies to a brand-new prospect's first-ever funding from
    this call.
    {_PAUSE_MARKER_NOTE}
    RULES:
    - Porter Capital is who Aiva works for — always say "Porter Capital"
      naturally if you reference who you're calling from. Never
      genericize it into something like "a financial services company."
    - The PROSPECT'S/LEAD'S company name is different — that one is
      internal context only, never spoken aloud to them.
    - Keep it tight and conversational. No rambling.
    - Never ask about their role, title, or job function — that doesn't
      serve qualification and just adds an unnecessary turn. Go straight
      from their check-in response into the pitch.
    - The check-in question is soft ("worth a quick chat?"), never a
      qualifying question ("what type of business are you in?").
    - NEVER quote rates, percentages, or dollar amounts.

    When they respond, call the pitch_result tool — do not write
    it out as text. Speak only the pitch and the check-in question;
    the tool call itself is a separate, silent action, never part
    of what you say out loud.

    [INTERNAL ROUTING LOGIC — never speak any of this, it only
    controls which tool arguments you pass]: if they respond
    positively, finish by calling the tool with result set to
    interested. If they show resistance or an objection, finish by
    calling the tool with result set to objection and put what they
    said in objection_text.

    If their response is unclear or doesn't clearly fit interested or
    objection — a mumbled answer, something off-topic, or anything you're
    not confident you understood correctly — do NOT guess which branch it
    is and do NOT call pitch_result yet. Ask ONE brief clarifying question
    naturally, e.g. "Sorry, was that a yes, or were you not so sure?", then
    follow whichever branch their answer now clearly fits. If it's STILL
    unclear after that one clarifying attempt, default to treating it as an
    objection — call pitch_result with result set to objection and
    objection_text noting the response was unclear, rather than forcing
    interested; it's safer to route to a human-guided objection-handling
    conversation than assume interest that wasn't really expressed, but
    never guess blindly.
    """


def _qualifying_block(company: str) -> str:
    return f"""
    CURRENT STAGE: QUALIFYING

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

    Follow-up template: "Got it — [natural reaction to their answer]. And
    do you guys currently factor any of your invoices, or handle that a
    different way?"

    If their answer to either the business-type question or the factoring
    question is unclear or doesn't clearly fit what you're asking — a
    mumbled answer, something off-topic, or anything you're not confident
    you understood correctly — do NOT guess and do NOT call
    qualifying_result yet. Ask ONE brief clarifying question naturally,
    then follow whichever branch their answer now clearly fits. If it's
    STILL unclear after that one clarifying attempt, stay in this stage
    and keep asking rather than guessing or completing the task — there's
    no urgency to end the call here, so never guess blindly just to move
    forward.

    ALREADY FACTORING WITH SOMEONE ELSE:
    This is NOT an automatic disqualifier — do not treat it as a
    dead end. A prospect who already has a factor may still be
    unhappy with them or open to switching, and there's a dedicated
    objection script for exactly this. If they say they already
    factor invoices with another company, call the qualifying_result
    tool with result set to objection and objection_text set to
    what they said — this hands off to the objection-handling flow
    (the "we already have a factor" script), not an instant
    not_qualified.
    Reserve not_qualified for when business type or factoring
    status genuinely rules them out for reasons OTHER than already
    having a factor (e.g. wrong type of business, or a clear
    negative signal unrelated to having an existing factor).

    OFFERING A ROUGH FUNDING ESTIMATE (only after business type and
    factoring status are established, and only for prospects who are NOT
    being routed to objection handling):

    Offer to give them a rough estimate — do not ask for their numbers
    upfront. Example:
    "I can actually give you a rough estimate of what we could get you, if
    that'd help — want me to?"

    If they say NO or decline:
    Skip the estimate entirely. Move directly to offering to connect them
    with a human advisor, same as usual.

    If they say YES:
    Ask for either their open accounts receivable (AR) balance or their
    annual revenue — whichever feels more natural, don't ask for both back
    to back.

    If the number they give you is unclear or garbled — a mumbled figure,
    or anything you're not confident you understood correctly — do NOT
    guess the number. Ask ONE brief clarifying question naturally, e.g.
    "Sorry, could you say that number again?", then use whichever number
    they now clearly give you. If it's STILL unclear after that one
    clarifying attempt, skip the estimate entirely and move to offering
    the advisor connection — same as if they'd said no to the estimate
    offer — rather than guessing at a number.

    Once you have ONE of these numbers, calculate a rough estimate:
    - If given an AR balance: estimate = that number x 90%
    - If given annual revenue: estimate = that number x 10%
    - If given both: use either one, your choice

    State this as a rough, non-binding estimate — always immediately
    followed by an offer to connect them with a human advisor for exact
    numbers. Example phrasing:
    "Based on that, we could likely get you up to around [estimate] — but
    the exact number really depends on the details, so let's get you
    connected with one of our advisors who can nail that down for you.
    Sound good?"

    NEVER present this estimate as a guaranteed or final number — it is
    always a rough approximation, and the human advisor always provides the
    real figure.

    If they agree to connect with an advisor (whether or not they took the
    estimate), move to booking — get their best day/time for a callback,
    same as the existing "open to exploring" flow.

    OFF-SCRIPT QUESTIONS (process, eligibility, rates):
    An engaged prospect asking questions about the process,
    eligibility, or rates is NOT a disqualifying signal — it's
    often a sign of genuine interest. Answer briefly, defer
    specifics to the human advisor, then return to qualifying or
    move toward booking if they seem ready. Do NOT call
    qualifying_result just because the conversation went off-script —
    only call it once you've actually determined qualified,
    not_qualified, or objection (see ALREADY FACTORING above) based
    on business type and factoring status (or a clear negative
    signal from the prospect).
    {_PAUSE_MARKER_NOTE}
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
    - NEVER quote rates, percentages, or dollar amounts EXCEPT the rough
      funding estimate above, which is the one sanctioned exception, and
      must always be framed as non-binding.
    - If asked about rates specifically (not the funding estimate), say
      only: "Honestly, it depends on a few things — how much you're
      invoicing, your customers, stuff like that. Rates typically run
      somewhere between 0.2 and 2 percent... but we'd build you an actual
      number once we know more."
    - If asked how fast funding happens: "Once you're set up with us,
      we move fast — usually under 48 hours from submitting an invoice."
    - One question at a time. Always.

    Use this knowledge to answer questions accurately:
    {PORTER_CAPITAL_KNOWLEDGE}

    When you have enough info, call the qualifying_result tool — do
    not write it out as text. Speak only your natural reply; the
    tool call itself is a separate, silent action, never part of
    what you say out loud.
    """


def _objection_block(company: str, objection_text: str) -> str:
    return f"""
    CURRENT STAGE: OBJECTION

    You are Aiva. The prospect just said: "{objection_text}"

    You are mid-call. The prospect already heard the ice breaker and
    AI disclosure from earlier in this conversation. Never repeat
    the opening greeting, never re-introduce yourself, never
    re-disclose being an AI unless directly asked again. Jump
    straight into your own job.

    {NATURAL_SPEECH_GUIDE}

    Follow this formula STRICTLY: Acknowledge -> Ask a question -> Continue.
    Never defend, never argue, never over-explain. One acknowledgment, one
    question, then stop.

    Examples:
    "We already have a factor" ->
    "That makes sense... most people do. Worth a quick look to see
     if we could actually do better for you?"

    "Not interested" ->
    "Totally fair. Can I ask real quick — is it bad timing, or just not
     something you need right now?"
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

    If their answer to "is it bad timing, or just not something you need
    right now?" is unclear or doesn't clearly fit either option — do NOT
    guess and do NOT call objection_result yet. Ask ONE brief clarifying
    rephrase naturally, then follow whichever result their answer now
    clearly fits. If it's STILL unclear after that one clarifying attempt,
    default to not_interested — conservative, since you shouldn't assume a
    callback was wanted if it wasn't clearly stated — but still end
    warmly, same as the existing not_interested pattern, rather than
    guessing blindly.

    "What are your rates" ->
    "Honestly, it depends on a few things — how much you're
     invoicing, your customers, stuff like that. Rates typically
     run somewhere between 0.2 and 2 percent... but we'd build you
     an actual number once we know more."

    "How fast could I get funded" ->
    "Once you're set up with us, we move fast — usually under 48
     hours from submitting an invoice."

    {_PAUSE_MARKER_NOTE}

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
      0.2-to-2-percent range shown in the script above. Never give a
      number more precise than that range.
    - Then call the objection_result tool — do not write it out as
      text. Speak only your acknowledgment and question; the tool
      call itself is a separate, silent action, never part of
      what you say out loud.
    """


def _booking_block() -> str:
    return f"""
    CURRENT STAGE: BOOKING

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

    If asked about rates: "Honestly, it depends on a few things —
    how much you're invoicing, your customers, stuff like that.
    Rates typically run somewhere between 0.2 and 2 percent... but
    we'd build you an actual number once we know more."

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

    Confirm back naturally, then call the booking_result tool — do
    not write it out as text. Speak only your natural confirmation;
    the tool call itself is a separate, silent action, never part
    of what you say out loud.

    RULES:
    - Never say the PROSPECT'S/LEAD'S company name back to them — it's
      internal context only, never spoken aloud. (Porter Capital,
      Aiva's own employer, is separate and fine to say if relevant.)
    - Never quote rates beyond the approved 0.2-to-2-percent range,
      and never promise approval
    - Keep it warm and brief — they already said yes
    """


def _disclosure_block() -> str:
    return f"""
    CURRENT STAGE: DISCLOSURE

    You are Aiva. Someone just asked if you are human or AI.

    You are mid-call. The prospect already heard the ice breaker
    earlier in this conversation. Never repeat the opening greeting
    or re-introduce yourself — just answer the question they just
    asked and jump straight into your own job.

    {NATURAL_SPEECH_GUIDE}

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
      wants_human, or wants_to_end — a mumbled answer, something off-topic,
      or anything you're not confident you understood correctly — do NOT
      guess and do NOT call disclosure_result yet. Ask ONE brief clarifying
      check-in naturally, e.g. "Are you good to keep chatting?", then
      follow whichever branch their answer now clearly fits. If it's STILL
      unclear after that one clarifying attempt, default to accepted —
      assume they're fine continuing — rather than guessing blindly.
    - Then call the disclosure_result tool — do not write it out as
      text. Speak only your natural reply; the tool call itself is
      a separate, silent action, never part of what you say out
      loud.
    """


def _exit_block() -> str:
    return f"""
    CURRENT STAGE: EXIT

    The prospect wants to be removed from our call list.

    YOUR ONLY JOB RIGHT NOW: Acknowledge warmly and end the call. Sound
    genuine — not robotic or over-apologetic.

    Example:
    "Of course — I'll take care of that right now. Thanks for letting me
     know, and have a good one."
    {_PAUSE_MARKER_NOTE}
    RULES:
    - Never say the prospect's company name back to them.
    - Speak ONLY the acknowledgment above. Never voice any conditional/meta
      text describing how to handle this situation.

    Then call opt_out — do not write it out as text.
    """


_STAGE_BLOCK_BUILDERS = {
    "opener": lambda self: _opener_block(
        self._company, self._city, self._contact_name
    ),
    "pitch": lambda self: _pitch_block(),
    "qualifying": lambda self: _qualifying_block(self._company),
    "objection": lambda self: _objection_block(self._company, self._objection_text),
    "booking": lambda self: _booking_block(),
    "disclosure": lambda self: _disclosure_block(),
    "exit": lambda self: _exit_block(),
}


# ============================================================
# THE SINGLE AGENT
# ============================================================

class Aiva(SpeechSafetyMixin, Agent):
    """One Agent, one ChatContext, for the entire call.

    self.current_stage is plain Python state (hello / opener / pitch /
    qualifying / objection / booking / disclosure / exit) — moving between
    stages is a state mutation + update_instructions() call, never a new
    Agent/AgentTask object and never a handoff. Digressions (an objection
    raised mid-pitch, a disclosure question mid-qualifying) save the stage
    they interrupted in self._return_stage and restore it afterward, using
    the model's real ChatContext to know what was already said instead of
    a task re-stating an assumption about it.
    """

    def __init__(self, lead, ctx: JobContext):
        self.lead = lead
        self.ctx = ctx
        self.call_result = "no_answer"
        self.current_stage = "hello"
        self._return_stage = None
        self._objection_text = ""
        self._call_ending = False  # guard against double _end_call()

        # Persisted alongside call_result at call-end (see when_call_starts).
        # objection_log accumulates every objection raised during the call
        # (a call can pass through _raise_objection more than once, distinct
        # from _objection_text above which only holds the CURRENT one for
        # the active objection stage's prompt), joined with "; " at write
        # time so earlier objections aren't lost when a later one overwrites
        # _objection_text.
        self.referral_details = None
        self.callback_details = None
        self.objection_log    = []

        self._company = lead.get("company_name", "the company")
        self._city = lead.get("city", "")
        self._contact_name = lead.get("contact_name") or ""
        self._state = lead.get("state", "")
        self._industry = lead.get("industry", "staffing")
        self._tier = lead.get("tier", "warm")

        super().__init__(instructions=self._build_instructions())

    # --------------------------------------------------------
    # Instructions are rebuilt fresh every stage transition and pushed via
    # update_instructions() — a real public SDK method (see
    # livekit.agents.voice.agent.Agent.update_instructions), designed to be
    # called repeatedly over a session's life. No handoff involved.
    # --------------------------------------------------------

    def _build_instructions(self) -> str:
        header = f"""
        You are Aiva, an AI sales assistant at Porter Capital, on a live
        outbound call.
        Lead: {self._company}, {self._city} {self._state}, {self._industry},
        Tier: {self._tier}

        {NATURAL_SPEECH_GUIDE}

        ABSOLUTE RULE: NEVER quote rates or numbers except the qualifying
        stage's explicitly sanctioned rough, non-binding funding estimate.

        This is ONE continuous conversation — you have full, real memory of
        everything said so far in the transcript above. Never assume
        something happened; check the actual conversation history above.
        Never repeat something you already said in this conversation.

        Only ever act on your CURRENT STAGE below. If at any point the
        prospect asks whether you're human/an AI, answer using the
        DISCLOSURE stage's script even if that's not your current stage,
        then continue exactly where you left off. If at any point they ask
        to be removed from the call list, switch to the EXIT stage instead.
        """
        stage_builder = _STAGE_BLOCK_BUILDERS.get(self.current_stage)
        stage_block = stage_builder(self) if stage_builder else ""
        return header + stage_block

    def _set_stage(self, stage: str):
        self.current_stage = stage
        return self.update_instructions(self._build_instructions())

    # --------------------------------------------------------
    # Call ending — same _end_call()/delete_room() pattern as agent.py.
    # No _finish_task chokepoint needed: there's only one Agent/activity for
    # the whole call, so there's no handoff for a stray reply to race
    # against. _call_ending is just a plain re-entrancy guard.
    # --------------------------------------------------------

    async def _end_call(self, closing_instructions: str | None = None):
        if self._call_ending:
            return
        self._call_ending = True
        if closing_instructions:
            await self.session.generate_reply(
                instructions=closing_instructions,
                allow_interruptions=False,
            )
        await self.ctx.delete_room()

    async def _end_call_bad_timing(self):
        self.call_result = "callback_later"
        await self._end_call(
            "Warmly say no problem at all, we'll try them again down the "
            "road, and thank them for their time. One or two short "
            "sentences."
        )

    # --------------------------------------------------------
    # HELLO — same manual asyncio.Event/wait_for escalation loop as
    # agent.py's OpenerTask.on_enter. Runs once, at session start.
    # --------------------------------------------------------

    async def on_enter(self):
        heard_speech = asyncio.Event()

        def _on_user_state(ev):
            if ev.new_state == "speaking":
                heard_speech.set()

        for attempt in range(OPENER_MAX_HELLO_ATTEMPTS):
            if self._call_ending:
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
                # They spoke — move to the opener stage and let the SDK's
                # default turn-completion flow generate the reply from the
                # now-updated instructions.
                await self._set_stage("opener")
                return
            except asyncio.TimeoutError:
                if attempt >= OPENER_MAX_HELLO_ATTEMPTS - 1:
                    self.call_result = "no_answer"
                    await self._end_call()
                    return
            finally:
                self.session.off("user_state_changed", _on_user_state)

    # --------------------------------------------------------
    # OPENER (TURN 1 -> RIGHT-PERSON CHECK -> TURN 2)
    # --------------------------------------------------------

    @function_tool()
    async def opener_result(self, result: str, referral_details: str = ""):
        """Call once the opener flow (TURN 1 through TURN 2) reaches a
        conclusion.
        result: interested / not_interested / bad_timing / wrong_number /
        hung_up / gatekeeper_referral.
        referral_details: the name/contact info they gave for the right
        person, only when result is gatekeeper_referral."""
        if self._call_ending:
            return "Call already ending."

        if result == "gatekeeper_referral":
            print(f"[Gatekeeper referral] {self._company}: {referral_details}")
            self.referral_details = referral_details
            self.call_result = "contacted"
            await self._end_call(
                "Acknowledge warmly, thank them for the info, and end the "
                'call in one short sentence — e.g. "Got it, thanks so much '
                '— have a great day!"'
            )
            return "Gatekeeper referral provided. Call ended."
        if result == "wrong_number":
            await self.session.generate_reply(
                instructions='Apologize briefly for the mix-up and add a warm '
                'sign-off — e.g. "Oh — sorry about that, my mistake. Have a '
                'good day." Keep it short, then stop.',
                allow_interruptions=False,
            )
            self.call_result = "not_interested"
            await self._end_call()
            return "Wrong number at company confirmation. Call ended."
        if result == "not_interested":
            self.call_result = "not_interested"
            await self._end_call(
                "Acknowledge warmly that's fine, and end the call in one "
                "short sentence."
            )
            return "Prospect declined. Call ended."
        if result == "bad_timing":
            await self._end_call_bad_timing()
            return "Bad timing at open. Marked callback_later. Call ended."
        if result == "hung_up":
            self.call_result = "no_answer"
            await self._end_call()
            return "Prospect hung up. Call ended."

        # interested
        self.call_result = "contacted"
        await self._set_stage("pitch")
        await self.session.generate_reply(
            instructions="Deliver the short pitch now, including one "
            "natural differentiator. Keep it tight and conversational, "
            "then ask the soft check-in question."
        )
        return "Opener done, interested. Pitch delivered."

    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    @function_tool()
    async def pitch_result(self, result: str, objection_text: str = ""):
        """result: interested / objection. objection_text: what they said,
        if result is objection."""
        if self._call_ending:
            return "Call already ending."
        if result == "objection":
            await self._raise_objection(objection_text)
            return "Objection raised from pitch."
        await self._set_stage("qualifying")
        await self.session.generate_reply(
            instructions="Start qualifying naturally. Acknowledge what they "
            "said first."
        )
        return "Pitch accepted. Qualifying started."

    # --------------------------------------------------------
    # QUALIFYING
    # --------------------------------------------------------

    @function_tool()
    async def qualifying_result(self, result: str, objection_text: str = ""):
        """result: qualified / not_qualified / objection.
        objection_text: what they said, only when result is objection
        (e.g. they already factor invoices with someone else)."""
        if self._call_ending:
            return "Call already ending."
        if result == "objection":
            await self._raise_objection(objection_text)
            return "Objection raised from qualifying."
        if result == "not_qualified":
            self.call_result = "not_interested"
            await self._end_call(
                "Acknowledge warmly, thank them for their time, and end the "
                "call in one short sentence."
            )
            return "Not qualified. Call ended."
        # qualified
        await self._set_stage("booking")
        await self.session.generate_reply(
            instructions="React warmly to their agreement. Ask for callback "
            "preference naturally, using the soft closing line about "
            "getting them set up with an advisor."
        )
        return "Qualified. Booking started."

    # --------------------------------------------------------
    # OBJECTION — a digression from pitch/qualifying/booking, not a
    # separate object. Saves whatever stage it interrupted and restores it
    # once resolved.
    # --------------------------------------------------------

    async def _raise_objection(self, objection_text: str):
        self._return_stage = self.current_stage
        self._objection_text = objection_text
        self.objection_log.append(objection_text)
        await self._set_stage("objection")
        await self.session.generate_reply(
            instructions="Acknowledge their objection in one short phrase, "
            "then ask exactly one question. Nothing else."
        )

    @function_tool()
    async def objection_result(self, result: str):
        """result: still_interested / not_interested / bad_timing /
        wants_callback"""
        if self._call_ending:
            return "Call already ending."
        if result == "not_interested":
            self.call_result = "not_interested"
            await self._end_call(
                "Acknowledge warmly, wish them well, and end the call in "
                "one short sentence."
            )
            return "Not interested. Call ended."
        if result == "bad_timing":
            await self._end_call_bad_timing()
            return "Bad timing. Marked callback_later. Call ended."
        if result == "wants_callback":
            await self._set_stage("booking")
            await self.session.generate_reply(
                instructions="React warmly to their agreement. Ask for "
                "callback preference naturally, using the soft closing line "
                "about getting them set up with an advisor."
            )
            return "Objection resolved to booking."
        # still_interested — resume whatever stage the objection interrupted
        await self._set_stage(self._return_stage or "qualifying")
        return f"Objection handled, still interested. Resumed {self.current_stage}."

    # --------------------------------------------------------
    # BOOKING
    # --------------------------------------------------------

    @function_tool()
    async def booking_result(self, callback_details: str):
        """Call when callback details are confirmed."""
        if self._call_ending:
            return "Call already ending."
        self.call_result = "callback_booked"
        self.callback_details = callback_details
        await self._end_call(
            "Confirm warmly that they're all set, referencing these "
            f"callback details naturally: {callback_details}. Say one of "
            "our advisors will call them then. Thank them for their time "
            "and say goodbye. Two sentences maximum."
        )
        return "Booking confirmed. Call ended."

    # --------------------------------------------------------
    # DISCLOSURE — a digression, same pattern as objection.
    # --------------------------------------------------------

    @function_tool()
    async def enter_disclosure(self):
        """Call when the prospect asks whether you are human or an AI, to
        switch to the disclosure script before answering."""
        if self._call_ending:
            return "Call already ending."
        self._return_stage = self.current_stage
        await self._set_stage("disclosure")
        await self.session.generate_reply(
            instructions="Answer honestly. Be warm and unbothered about "
            "being an AI."
        )
        return "Disclosure stage entered."

    @function_tool()
    async def disclosure_result(self, reaction: str):
        """reaction: accepted / wants_human / wants_to_end"""
        if self._call_ending:
            return "Call already ending."
        if reaction == "wants_human":
            await self._set_stage("booking")
            await self.session.generate_reply(
                instructions="React warmly to their agreement. Ask for "
                "callback preference naturally, using the soft closing line "
                "about getting them set up with an advisor."
            )
            return "Wants human. Booking started."
        if reaction == "wants_to_end":
            self.call_result = "not_interested"
            await self._end_call(
                "Acknowledge warmly that's fine, thank them, and end the "
                "call in one short sentence."
            )
            return "Prospect wants to end after disclosure. Call ended."
        # accepted — resume whatever stage disclosure interrupted
        await self._set_stage(self._return_stage or "opener")
        return f"Disclosure accepted. Resumed {self.current_stage}."

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    @function_tool()
    async def enter_exit(self):
        """Call when the prospect asks to be removed from the call list /
        stop calling, before acknowledging."""
        if self._call_ending:
            return "Call already ending."
        await self._set_stage("exit")
        await self.session.generate_reply(
            instructions="Acknowledge naturally and warmly, with a brief "
            "thanks before the goodbye. Two sentences maximum."
        )
        return "Exit stage entered."

    @function_tool()
    async def opt_out(self):
        """Call immediately after acknowledging the opt-out request."""
        if self._call_ending:
            return "Call already ending."
        if os.getenv("TEST_MODE", "false").lower() == "true":
            print(
                "TEST MODE — suppression_list NOT updated. Would have added: "
                f"company_name={self.lead.get('company_name', '')!r} "
                f"website_domain={self.lead.get('website_domain', '')!r} reason='opted_out'"
            )
        else:
            add_to_suppression(
                company_name=self.lead.get("company_name", ""),
                website_domain=self.lead.get("website_domain", ""),
                reason="opted_out",
            )
        self.call_result = "suppressed"
        await self._end_call()
        return "Opted out. Suppression logged. Call ended."


# ============================================================
# LLM SELECTION — identical to agent.py
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


def _build_llm(provider: str):
    if provider == "groq":
        return groq.LLM(model="llama-3.3-70b-versatile", timeout=10.0)
    elif provider == "grok":
        grok_model = os.getenv("GROK_MODEL", "xai/grok-4-1-fast-non-reasoning")
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            return openai_plugin.LLM(
                model=grok_model.removeprefix("xai/"),
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
                timeout=10.0,
            )
        return inference.LLM(model=grok_model)
    elif provider == "claude":
        return anthropic.LLM(model="claude-haiku-4-5")
    elif provider == "gpt":
        return openai_plugin.LLM(model="gpt-4o-mini")
    elif provider == "qwen":
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
        return openai_plugin.LLM.with_ollama(model="gemma4:latest", base_url=OLLAMA_BASE_URL)
    elif provider == "gptoss":
        return openai_plugin.LLM.with_ollama(model="llama3.1:latest", base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print(f"=== [{AGENT_VERSION}] Testing LLM: {provider.upper()} ===")

    primary = _build_llm(provider)

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
# TTS SELECTION — identical to agent.py
# ============================================================

KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
CHATTERBOX_BASE_URL = os.getenv("CHATTERBOX_BASE_URL", "http://localhost:8880/v1")


def get_tts():
    provider = os.getenv("TTS_PROVIDER", "cartesia").lower()
    print(f"=== [{AGENT_VERSION}] Testing TTS: {provider.upper()} ===")

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
# STARTUP — identical to agent.py
# ============================================================

def prewarm_qwen(proc: JobProcess) -> None:
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
    print(f"[{AGENT_VERSION}] Calling: {lead['company_name']} | Tier: {lead['tier']}")

    call_id = await asyncio.to_thread(
        create_call,
        lead["lead_candidate_id"],
        "v2",
        os.getenv("LLM_PROVIDER", "groq").lower(),
        os.getenv("TTS_PROVIDER", "cartesia").lower(),
        ctx.job.room.sid or ctx.room.sid or ctx.room.name,
    )

    pipeline = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=get_llm(),
        tts=get_tts(),
        vad=silero.VAD.load(),
    )

    aiva = Aiva(lead, ctx)

    transcript = CallTranscriptRecorder(call_id, lambda: aiva.current_stage)
    pipeline.on(
        "conversation_item_added",
        transcript.on_conversation_item_added,
    )

    call_ended = asyncio.Event()
    pipeline.on("close", lambda ev: call_ended.set())

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
            room=ctx.room,
            agent=aiva,
            room_input_options=RoomInputOptions(),
        )

        # aiva.on_enter() (the hello-escalation loop) runs automatically as part
        # of pipeline.start() — no generate_reply()/tool-call indirection needed
        # to reach it, since there's no AgentTask.__await_impl() inline_task
        # requirement here: on_enter is just a normal method on the one Agent
        # this session ever uses.

        await call_ended.wait()
    finally:
        await transcript.close(aiva.call_result)

    objection_text = "; ".join(aiva.objection_log) if aiva.objection_log else None

    if os.getenv("TEST_MODE", "false").lower() == "true":
        print(
            f"TEST MODE — lead status NOT updated. Would have set: {aiva.call_result} | "
            f"referral_details={aiva.referral_details!r} | "
            f"callback_details={aiva.callback_details!r} | "
            f"objection_text={objection_text!r}"
        )
    else:
        update_lead_status(
            lead_candidate_id=lead["lead_candidate_id"],
            new_status=aiva.call_result,
            referral_details=aiva.referral_details,
            callback_details=aiva.callback_details,
            objection_text=objection_text,
        )

    print(f"[{AGENT_VERSION}] Call ended. Result: {aiva.call_result}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=when_call_starts,
        prewarm_fnc=prewarm_qwen,
        job_executor_type=JobExecutorType.PROCESS,
        num_idle_processes=1,
        initialize_process_timeout=40.0,
    ))
