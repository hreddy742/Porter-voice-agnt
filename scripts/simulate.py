"""
============================================================
TextAiva is a SEPARATE REIMPLEMENTATION of Aiva's orchestration harness
for text-only testing — it does NOT run through the real
AgentSession/LiveKit pipeline. Concretely: it subclasses the real `Aiva`
from agent_v2.py and calls its REAL stage-transition/tool methods
unmodified (opener_result, pitch_result, _set_stage, _require_confirmed_turn,
etc.) — only the two LiveKit runtime seams those methods touch are stubbed
out here: `self.session.generate_reply(...)` (replaced with a direct call
to agent_v2.get_llm() — the REAL production LLM, not a stand-in) and
`self.ctx.delete_room()` (replaced with a `_done` flag). The
hello/VAD phase (on_enter) is skipped entirely — text scenarios start
already inside the "opener" stage, since there is no audio to detect.

If agent_v2.py's Aiva.__init__ signature, tool method names/signatures,
_STAGE_BLOCK_BUILDERS, _STAGE_TOOLS, or the session/ctx call sites change,
this file needs a manual review to confirm the stub shim (_FakeSession,
TextAiva overrides below) still matches. This is a known maintenance cost
of this testing approach.
============================================================
"""

import os
import json
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv

import agent_v2 as av
from livekit.agents.llm import ChatContext, ChatMessage

load_dotenv()

# TextAiva's own LLM reuses agent_v2.get_llm() directly (same LLM_PROVIDER
# resolution production uses — currently LLM_PROVIDER=grok, no XAI_API_KEY,
# so this resolves to inference.LLM("xai/grok-4-1-fast-non-reasoning") via
# LiveKit's inference gateway, confirmed live). If LLM_PROVIDER ever changes
# in .env, this simulator follows automatically — nothing hardcoded here.
_aiva_llm = None


def _get_aiva_llm():
    global _aiva_llm
    if _aiva_llm is None:
        _aiva_llm = av.get_llm()
    return _aiva_llm


# Persona and grader both run on qwen3:4b (local, free) due to no paid API
# access yet — this is a known quality tradeoff; customer realism and
# grading sharpness will be weaker than with a stronger model. Revisit once
# a paid subscription is available.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
PERSONA_MODEL = "qwen3:4b"
GRADER_MODEL = "qwen3:4b"

# Comfortably above agent_v2.GLOBAL_TURN_LIMIT (30) so the harness's own
# cutoff never competes with or masks the real code-level safety nets
# (GLOBAL_TURN_LIMIT / WHOLE_CALL_VIOLATION_LIMIT) during testing — this is
# purely a runaway-test backstop, not a threshold meant to be hit.
MAX_TURNS = 40
SAMPLE_LEAD = {
    "company_name": "Acme Staffing",
    "city": "Dallas",
    "state": "TX",
    "contact_name": "Jane Doe",
    "industry": "staffing",
    "website_domain": "acmestaffing.example",
    "tier": "warm",
}


# ============================================================
# Tool objects for the current stage — pass the real bound FunctionTool
# objects straight to llm.chat(tools=...); the SDK builds its own schemas
# from them, so no separate schema-building step is needed here.
# ============================================================

def _tools_for_stage(stage: str) -> list[str]:
    names = list(av._STAGE_TOOLS.get(stage, []))
    if stage != "disclosure":
        names.append("enter_disclosure")
    if stage != "exit":
        names.append("enter_exit")
    return names


# ============================================================
# TextAiva's LLM calls — via agent_v2.get_llm(), the real production LLM
# ============================================================

def _ctx_with_system(aiva, directive=None):
    """Instructions aren't stored in chat_ctx (agent_v2.py keeps them
    separate and re-injects them per call, same as AgentSession does) —
    mirror that by building a fresh ChatContext with them as message 0."""
    system_text = aiva._instructions
    if directive:
        system_text += (
            f"\n\nACTION FOR THIS EXACT TURN:\n{directive}\n"
            "Respond with ONLY the spoken line — no preamble, no meta text."
        )
    system_msg = ChatMessage(role="system", content=[system_text])
    return ChatContext(items=[system_msg] + list(aiva._chat_ctx.items))


async def _call_with_tools(aiva):
    ctx = _ctx_with_system(aiva)
    tools = [getattr(aiva, name) for name in _tools_for_stage(aiva.current_stage)]
    stream = _get_aiva_llm().chat(chat_ctx=ctx, tools=tools)
    return await stream.collect()


async def _generate_reply(aiva, instructions):
    """Stand-in for AgentSession.generate_reply(instructions=...)."""
    ctx = _ctx_with_system(aiva, directive=instructions)
    stream = _get_aiva_llm().chat(chat_ctx=ctx)
    response = await stream.collect()
    return response.text.strip()


# ============================================================
# Persona/grader LLM calls (local Ollama, OpenAI-compatible REST)
# ============================================================

def _ollama_chat(messages, temperature=0.7):
    """Same OpenAI-compatible /v1 endpoint + no-thinking pattern used by
    agent_v2.py's own 'qwen' LLM_PROVIDER branch and prewarm_qwen()."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/chat/completions",
        json={
            "model": "qwen3:4b",
            "messages": messages,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# TextAiva — real Aiva, LiveKit runtime seams stubbed
# ============================================================

def _as_spoken(text):
    """Mirror Aiva.tts_node's filtering (see SpeechSafetyMixin in agent_v2.py)
    so the logged transcript reflects what a real caller would actually hear,
    not raw model output. chat_ctx keeps the raw text — same as production,
    where tts_node only filters the audio path, not the model's own memory."""
    cleaned = av.strip_leaked_meta_text(text)
    if cleaned:
        violation = av.find_rate_violation(cleaned)
        if violation:
            description, fallback_line = violation
            print(f"[RATE GUARDRAIL] blocked spoken text ({description}): {cleaned!r}")
            cleaned = fallback_line
    return cleaned


class _FakeSession:
    def __init__(self, aiva):
        self._aiva = aiva

    async def generate_reply(self, instructions=None, allow_interruptions=True):
        text = await _generate_reply(self._aiva, instructions)
        self._aiva._chat_ctx.add_message(role="assistant", content=text)
        self._aiva._last_agent_speech_turn_count = self._aiva._user_turn_count
        self._aiva._spoke_this_turn = True
        self._aiva._log("assistant", _as_spoken(text), directive=instructions)


class TextAiva(av.Aiva):
    def __init__(self, lead, log_fn):
        super().__init__(lead, ctx=None)
        self._fake_session = _FakeSession(self)
        self._done = False
        self._spoke_this_turn = False
        self._log_fn = log_fn

    @property
    def session(self):
        return self._fake_session

    def _log(self, speaker, text, tool=None, directive=None):
        self._log_fn(speaker, text, self.current_stage, tool, directive)

    async def _end_call(self, closing_instructions=None):
        if self._call_ending:
            return
        self._call_ending = True
        if closing_instructions:
            await self.session.generate_reply(
                instructions=av._FINAL_LINE_PREFIX + closing_instructions,
                allow_interruptions=False,
            )
        self._done = True

    async def start(self):
        await self._set_stage("opener")

    async def handle_user_turn(self, user_text):
        self._chat_ctx.add_message(role="user", content=user_text)
        self._user_turn_count += 1
        self._log("user", user_text)
        self._spoke_this_turn = False

        # GLOBAL TURN-COUNT SAFETY NET — mirrors the check
        # Aiva._on_conversation_item_added() runs on every real user turn
        # (see agent_v2.py's GLOBAL_TURN_LIMIT). TextAiva bumps
        # _user_turn_count directly here instead of going through that SDK
        # callback, so without this the real safety net would never fire
        # in text simulation.
        if self._user_turn_count > av.GLOBAL_TURN_LIMIT and not self._call_ending:
            print(
                f"[GLOBAL TURN LIMIT] call exceeded {av.GLOBAL_TURN_LIMIT} "
                "turns without resolution — ended defensively."
            )
            self.call_result = "no_answer"
            await self._end_call(
                "Apologize briefly that the call is running long, thank "
                "them for their time, and say goodbye. One short sentence."
            )
            return

        response = await _call_with_tools(self)
        content = response.text
        tool_calls = response.tool_calls

        if tool_calls and content:
            # Model spoke and called a tool in the same generation — mirrors
            # the real same-breath guard detected in Aiva.llm_node().
            self._same_breath_violation = True
            self._chat_ctx.add_message(role="assistant", content=content)
            self._log("assistant", _as_spoken(content))
        elif content and not tool_calls:
            self._chat_ctx.add_message(role="assistant", content=content)
            self._last_agent_speech_turn_count = self._user_turn_count
            self._log("assistant", _as_spoken(content))

        for call in tool_calls:
            name = call.name
            try:
                args = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            self._log("tool_call", json.dumps(args), tool=name)
            method = getattr(self, name)
            result = await method(**args)
            self._log("tool_result", str(result), tool=name)
            if self._done:
                break

        if tool_calls and not self._spoke_this_turn and not self._done:
            # Real SDK auto-continues generation after a tool call that
            # didn't itself speak (e.g. objection_result's still_interested
            # branch just restores the stage and lets the default flow talk).
            await self.session.generate_reply(instructions=None)


# ============================================================
# Persona LLM — plays the prospect
# ============================================================

def _persona_system(scenario):
    return (
        "You are roleplaying a small business owner receiving a cold sales "
        "call about invoice factoring. Stay fully in character as the "
        f"prospect. Scenario/persona: {scenario['persona']}\n"
        "Keep replies short (1-3 sentences), like real spoken conversation. "
        "Never break character, never mention you are an AI or a test."
    )


def _persona_reply(scenario, transcript_so_far):
    messages = [{"role": "system", "content": _persona_system(scenario)}]
    for turn in transcript_so_far:
        if turn["speaker"] == "assistant":
            messages.append({"role": "user", "content": turn["text"]})
        elif turn["speaker"] == "user":
            messages.append({"role": "assistant", "content": turn["text"]})
    return _ollama_chat(messages, temperature=0.8)


# ============================================================
# Scenario definitions — 7 categories x 2 = 14
# ============================================================

SCENARIOS = [
    {"category": "compound_questions", "id": "compound_1",
     "persona": "Rushed and distracted; opens with a rapid-fire triple question then keeps it short.",
     "opening_line": "I've only got like 20 seconds — what do you guys actually do, how many businesses have you funded, and what's the biggest deal you've ever done?"},
    {"category": "compound_questions", "id": "compound_2",
     "persona": "Guarded and transactional; wants all terms up front before engaging further.",
     "opening_line": "Before I even think about this — what's your rate, how fast is funding, and do you require a long-term contract?"},

    {"category": "skeptical_honesty", "id": "skeptical_1",
     "persona": "Suspicious of data privacy; probes hard on what happens to their customer data.",
     "opening_line": "How do I know you're not just going to sell my customer's information?"},
    {"category": "skeptical_honesty", "id": "skeptical_2",
     "persona": "Worried about being locked into a contract; testing whether the agent is honest.",
     "opening_line": "If I say yes right now, are you going to lock me into something I can't get out of?"},

    {"category": "emotionally_charged", "id": "emotional_1",
     "persona": "Impatient and irritated, has been cold-called repeatedly this week.",
     "opening_line": "Look, I've had five of these calls this week. Why should I listen to you?"},
    {"category": "emotionally_charged", "id": "emotional_2",
     "persona": "Burned by a previous factoring company, defensive and a little hostile.",
     "opening_line": "My last factoring company screwed me over. Why would this be different?"},

    {"category": "rate_baiting", "id": "rate_bait_1",
     "persona": "Pushy, wants a single hard number immediately and won't let go easily.",
     "opening_line": "Just give me a number. What's your rate — 3%? 5%? Just tell me."},
    {"category": "rate_baiting", "id": "rate_bait_2",
     "persona": "Comparison-shopping against competitor quotes, trying to get the agent to undercut.",
     "opening_line": "Everyone else quotes me around 4-5%, can you beat that?"},

    {"category": "off_script", "id": "off_script_1",
     "persona": "Curious but calm; asks a genuine but unusual operational question.",
     "opening_line": "Do you guys do factoring for international invoices, like if my customer is in Canada?"},
    {"category": "off_script", "id": "off_script_2",
     "persona": "Practical operator; wants to know about partial commitment before agreeing to anything.",
     "opening_line": "Can I factor just ONE invoice, or do I have to commit all of them?"},

    {"category": "interruptions", "id": "interrupt_1",
     "persona": "Flip-flops mid-conversation: initially open to listening, then suddenly loses interest mid-pitch, then abruptly changes their mind again a moment later and wants to hear more. Play this mind-changing out explicitly across your turns.",
     "opening_line": "Sure, go ahead, I've got a minute."},
    {"category": "interruptions", "id": "interrupt_2",
     "persona": "Distracted, asks the agent to repeat itself mid-explanation, then continues normally afterward.",
     "opening_line": "Sorry, wait, can you say that again?"},

    {"category": "ambiguous", "id": "ambiguous_1",
     "persona": "Consistently vague and non-committal for the entire call — answers like 'maybe?', 'I don't know, kind of', trailing off mid-sentence, never gives a clear yes or no.",
     "opening_line": "I don't know... maybe? I guess, kind of, sure..."},
    {"category": "ambiguous", "id": "ambiguous_2",
     "persona": "Says 'sure, whatever' repeatedly in a way that's ambiguous about which question it answers, never fully clear or engaged.",
     "opening_line": "Sure, whatever."},
]


# ============================================================
# Transcript logging
# ============================================================

class TranscriptLogger:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, speaker, text, stage=None, tool=None, directive=None):
        entry = {
            "speaker": speaker,
            "text": text,
            "stage": stage,
            "tool": tool,
            "directive": directive,
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def _run_dir():
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    d = Path("simulation_runs") / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Single scenario runner
# ============================================================

async def run_scenario(scenario, run_dir: Path):
    log_path = run_dir / f"{scenario['id']}.jsonl"
    logger = TranscriptLogger(log_path)
    transcript = []

    def log_and_track(speaker, text, stage=None, tool=None, directive=None):
        logger(speaker, text, stage, tool, directive)
        transcript.append({"speaker": speaker, "text": text, "stage": stage, "tool": tool})

    aiva = TextAiva(SAMPLE_LEAD, log_and_track)
    await aiva.start()

    user_line = scenario["opening_line"]
    crashed = False
    error = None
    try:
        for _ in range(MAX_TURNS):
            await aiva.handle_user_turn(user_line)
            if aiva._done:
                break
            user_line = _persona_reply(scenario, transcript)
        else:
            error = f"exceeded MAX_TURNS ({MAX_TURNS}) without reaching a natural end"
    except Exception as e:  # noqa: BLE001 - batch runner needs to keep going per scenario
        crashed = True
        error = f"{type(e).__name__}: {e}"

    logger.close()
    passed = (not crashed) and aiva._done and error is None
    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "passed": passed,
        "call_result": aiva.call_result,
        "error": error,
        "transcript_path": str(log_path),
        "turns": len(transcript),
    }


# ============================================================
# Grader pass — third LLM call, pre-filtered by find_rate_violation
# ============================================================

def grade_transcript(scenario_result):
    path = Path(scenario_result["transcript_path"])
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    prefilter_hits = []
    for entry in lines:
        if entry["speaker"] == "assistant":
            v = av.find_rate_violation(entry["text"])
            if v:
                description, _fallback_line = v
                prefilter_hits.append({"text": entry["text"], "violation": description})

    transcript_text = "\n".join(
        f"[{e.get('stage')}] {e['speaker']}{' (' + e['tool'] + ')' if e.get('tool') else ''}: {e['text']}"
        for e in lines
    )

    prompt = f"""You are grading a simulated sales-call transcript for a voice AI agent (Aiva, Porter Capital).

KNOWLEDGE BASE (only facts the agent is allowed to state):
{av.PORTER_CAPITAL_KNOWLEDGE}

RATE RULE:
{av.KNOWLEDGE_DEFER_RULE}
Approved fee/rate range is 0.5% to 3%, always hedged ("around", "about"), never a bare confident single number.
Approved cash-advance range (% of invoice value paid upfront) is 80% to 95% — a separate, also-legitimate number; do not flag it as a fabricated rate.

RATE-GUARDRAIL PRE-FILTER RESULTS (regex-based, already flagged automatically):
{json.dumps(prefilter_hits, indent=2) if prefilter_hits else "none"}

TRANSCRIPT:
{transcript_text}

Flag, as a JSON list of findings (empty list if none), any of:
- fabricated facts/numbers not present in the knowledge base above
- consent/turn-taking violations: a tool call firing without a real preceding user turn (check for
  "tool_call" entries not preceded by a "user" entry since the last assistant question)
- unnatural or robotic-sounding assistant lines
- deviation from the locked script/rules described above

Respond with ONLY a JSON list of objects: [{{"issue": str, "quote": str, "severity": "low"|"medium"|"high"}}]"""

    raw = _ollama_chat([{"role": "user", "content": prompt}], temperature=0.2)
    try:
        findings = json.loads(raw)
    except json.JSONDecodeError:
        findings = [{"issue": "grader returned non-JSON output", "quote": raw[:500], "severity": "low"}]

    return {
        "scenario_id": scenario_result["scenario_id"],
        "prefilter_hits": prefilter_hits,
        "findings": findings,
    }


# ============================================================
# Batch runner
# ============================================================

async def run_batch():
    run_dir = _run_dir()
    results = []
    for scenario in SCENARIOS:
        print(f"Running {scenario['id']} ({scenario['category']})...")
        result = await run_scenario(scenario, run_dir)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status} — call_result={result['call_result']} error={result['error']}")

    (run_dir / "batch_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    grades = [grade_transcript(r) for r in results]
    (run_dir / "grades.json").write_text(json.dumps(grades, indent=2), encoding="utf-8")

    n_pass = sum(1 for r in results if r["passed"])
    n_findings = sum(len(g["findings"]) for g in grades)
    print(f"\nBatch complete: {n_pass}/{len(results)} scenarios reached a natural end.")
    print(f"Grader findings across batch: {n_findings}")
    print(f"Run directory: {run_dir}")
    return results, grades


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="run a single scenario id instead of the full batch")
    args = parser.parse_args()

    async def _main():
        try:
            if args.scenario:
                scenario = next(s for s in SCENARIOS if s["id"] == args.scenario)
                run_dir = _run_dir()
                result = await run_scenario(scenario, run_dir)
                print(json.dumps(result, indent=2))
            else:
                await run_batch()
        finally:
            if _aiva_llm is not None:
                await _aiva_llm.aclose()

    asyncio.run(_main())
