# Upgrade Tracker

## Per-turn endpoint latency is visible before tuning (2026-07-22)
The historical transcript showed a delayed user-turn commit but could not
distinguish endpoint/VAD delay from transcription delay. LiveKit 1.6.4 already
attaches both measurements to each committed user `ChatMessage`; `agent_v2.py`
now prints them as one `[TURN LATENCY]` line with the active stage. This is
read-only instrumentation and does not change VAD, STT, endpointing, or reply
behavior. Endpoint settings remain unchanged until a real phone call supplies
the missing evidence.

## Deterministic stage transitions no longer make a second LLM call (2026-07-22)
The interested/qualified/callback tool branches already knew both the next
stage and its opening sentence, but they asked the LLM to generate that
sentence again. Each affected turn therefore waited for two sequential LLM
calls before TTS could start.

Those five branches now use LiveKit `say()` with reviewed, interruptible
opening lines for pitch, qualifying, and booking. Dynamic questions,
objections, answers, and closing decisions still use the LLM. Every function
tool still returns `None`, preserving the single-response-owner fix.

Validation against the installed LiveKit 1.6.4 stack and GPT-4o-mini produced
exactly one pitch after one tool call. Three new transition samples averaged
1.153 seconds through the LLM/tool path. Combined with the separately measured
Cartesia first-audio average of 0.371 seconds, estimated time to first audio is
about 1.52 seconds, down from the prior measured 3.12-second two-LLM path
(about 1.60 seconds / 51% lower), excluding endpoint detection.

## Critical opener wording is deterministic (2026-07-21)
The opener's compliance/state-control language no longer depends on model
wording. When the model reaches the cold-call/AI disclosure, `llm_node()`
substitutes one exact reviewed sentence before text reaches transcription or
TTS. The `needs_clarification` tool path uses LiveKit `say()` with one exact
clarification sentence, so it cannot paraphrase, omit the AI disclosure, or
accidentally pitch.

The real GPT/LiveKit opener diagnostic asserts exact recorded assistant text
for both lines and passed, alongside the interested and not-interested state
transitions. Ordinary pitch/qualifying wording remains intentionally dynamic;
only business-critical control language is locked.

## Opener must resolve after the AI/time disclosure (2026-07-21)
Call 27 exposed that the prompt alone did not guarantee stage progression:
after Aiva delivered the cold-call/AI disclosure, GPT spoke a pitch without
calling `opener_result`, leaving all 27 persisted turns incorrectly stamped
as `opener`.

The agent now arms an opener-result requirement only after that disclosure is
actually generated. LiveKit is given a named `opener_result` tool choice only
when a newer user message is present, so silence/background generations cannot
classify the prospect. `result` is a constrained enum, including the safe
`needs_clarification` outcome; that outcome rephrases the disclosure and stays
in opener instead of guessing.

Real GPT/LiveKit validation covered all three critical branches:
- "I just got ten seconds" -> `interested`, then `opener -> pitch`;
- "What do you mean?" -> `needs_clarification`, remains in opener;
- clear "not interested" -> `not_interested`, closes without pitching.

## Explicit human-callback intent bypasses stale stages (2026-07-21)
Production call 27 remained in `opener` while the conversation improvised into
rates and factoring. When the prospect explicitly asked to set up a call with
a sales rep/advisor, stale opener instructions interpreted that as a
gatekeeper referral and ended the call with `final_call_result='contacted'`.

`agent_v2.py` now recognizes a narrow set of explicit scheduling/human-advisor
requests before normal inference and moves directly to `booking`. The same
natural response uses booking's tools and asks for callback number/day/time;
it cannot ask for the "right person" or take the referral-ending branch.

Validated with a deterministic regression using call 27's exact wording,
negative and idempotency cases, and a real GPT/LiveKit run starting from a
deliberately stale opener. The live run transitioned `opener -> booking`,
produced exactly one callback-details question, and did not end the call.

## Drift correction deferred to the next user turn (2026-07-21)
Live call 25 proved the old forced correction was unsafe. On the third
no-tool assistant turn it started a background `generate_reply()` immediately,
before the prospect answered the question just asked. In qualifying this
created a second assistant reply; in objection it forced a guessed
`not_interested` result and ended an interested call.

The guard now records a pending correction instead of generating anything.
`llm_node()` applies one ephemeral, stage-aware instruction only when its
`chat_ctx` contains a newer user message. The instruction asks the model to
call the stage tool when the new answer is sufficient, or ask one specific
clarifying question when it is not. It does not force `tool_choice`, roll back
turn stamps, or create a second response owner.

Validated three ways:
- deterministic qualifying and objection regressions reproduce the exact
  three-turn threshold and prove no task/reply is created;
- the correction does not apply without a newer user message and is consumed
  exactly once when that message arrives;
- the real GPT/LiveKit diagnostic passed both stages: no unsolicited reply
  after the threshold, and at most one assistant reply on the next user turn.

`NO_TOOL_DRIFT_LIMIT`, the whole-call violation counter, and
`GLOBAL_TURN_LIMIT` remain unchanged as independent backstops.

## Same-breath fix confirmed final; LiveKit "structured output" does not apply (2026-07-21)
Investigated replacing the llm_node() buffer-and-discard same-breath fix
(agent_v2.py, narration+tool-call bundling guard) with LiveKit's documented
structured-output pattern (`response_format` + `process_structured_output`,
docs.livekit.io/agents/logic/tools/definition/). Conclusion: does not apply,
no further investigation needed on this specific question.

- `response_format` only constrains the LLM's text/content channel to a JSON
  schema (e.g. splitting `voice_instructions` from `response`). It has no
  effect on the `tool_calls` channel — nothing stops a model from emitting
  narrated content and a tool call in the same generation while under
  `response_format`, which is exactly the bug the buffer-and-discard fix
  exists to catch. Confirmed against the actual (now-deleted, pulled from
  git history) `examples/voice_agents/structured_output.py`: its own
  `llm_node` forwards `tool_calls` deltas completely unfiltered.
- This holds regardless of which LLM providers are in the fallback chain
  (verified for both the old Anthropic-inclusive chain and the current
  OpenAI/Groq-only chain below) — the reason it doesn't apply is that
  `response_format` and `tool_calls` are orthogonal channels on every
  provider's API, not a provider-support gap.
- Also checked: `ToolFlag.IGNORE_ON_ENTER` (real, confirmed in installed
  1.6.4) is redundant with our existing per-stage `_STAGE_TOOLS` gating —
  `opener_result` etc. are already unregistered during `on_enter()`'s hello
  loop via `_STAGE_TOOLS["hello"] = []`, which is broader than
  `IGNORE_ON_ENTER`'s on_enter-only scope. No action taken.
- `_require_confirmed_turn()`, the violation ladder, and the rate guardrail
  all remain as-is — nothing above replaces or makes any of them redundant.

## LLM provider: dropped Anthropic, OpenAI primary / Groq fallback (2026-07-21)
Removed `anthropic.LLM` entirely from `_build_llm()`/`get_llm()` in
agent_v2.py (import + the `"claude"` provider branch) — it was the only
Anthropic reference in the file. `.env` now sets `LLM_PROVIDER=gpt`,
`LLM_FAILOVER_PROVIDER=groq`. The `get_llm()` failover gate had been
hardcoded to only build a `FallbackAdapter` when `LLM_PROVIDER=groq`
specifically; generalized it to build the fallback whenever
`LLM_FAILOVER_PROVIDER` is set and differs from the primary, so
OpenAI-primary/Groq-fallback (and any other combination) works, not just
Groq-primary.

## LLM failover provider (Groq -> local qwen3:4b)
Current LLM failover is local qwen3:4b via Ollama — noticeably weaker
reasoning than Groq's 70B primary. This exists purely to prevent dead
silence during a Groq outage/rate-limit, not to maintain conversation
quality. Revisit once a paid Groq tier or other reliable cloud fallback
is available.

- qwen3:4b defaults to an internal "thinking" mode that generates a long
  hidden reasoning chain before any tool call/reply — measured ~45s for a
  single tool call, which is indistinguishable from dead air on a live
  call. Fixed by passing `extra_body={"chat_template_kwargs":
  {"enable_thinking": False}}` (agent.py `_build_llm("qwen")`), which
  drops warm-model latency to ~5s.
- Known remaining latency risk: Ollama unloads an idle model from memory
  after its default keep_alive window. A "cold" first call (model not
  currently loaded) measured ~24s before the tool call — still much
  better than the ~45s thinking-mode hang, but still a noticeable pause
  on a live call. Mitigated: `prewarm_qwen()` (agent.py / agent_v2.py)
  runs as `WorkerOptions.prewarm_fnc`, pinging the qwen model once at
  worker startup, combined with `OLLAMA_KEEP_ALIVE` so the model stays
  resident between calls instead of unloading on its default window.

## livekit-rtc webrtc-sys panic fix
- Bug: github.com/livekit/rust-sdks/issues/944
- Fix merged: rust-sdks PR #1098 (2026-07-09)
- Current livekit-agents (1.6.4) still pins livekit==1.1.12,
  which predates the fix
- Workaround applied: job_executor_type=JobExecutorType.PROCESS
  in agent.py (isolates crash to single call, not whole worker)
- TODO: once a livekit-agents release bundles rust-sdks with the
  fix (check pip index versions livekit for anything past 1.1.12
  built after 2026-07-09), upgrade and remove/reconsider the
  PROCESS workaround if THREAD mode is preferred for performance

- CAVEAT confirmed 2026-07-13: the PROCESS workaround does NOT protect
  local Playground/console testing (`python agent.py console`). LiveKit's
  own CLI hardcodes `server._job_executor_type = JobExecutorType.THREAD`
  for that path (livekit/agents/cli/cli.py, `_run_tcp_console`), overriding
  whatever WorkerOptions configures. Under THREAD, the job shares the
  worker's process, so this same webrtc-sys panic takes the whole worker
  down instead of being contained to one call — exactly what happened
  during this session's testing. This is expected, pre-existing SDK
  behavior tied to console-mode testing, not a regression introduced by
  any agent.py change. The PROCESS isolation is real and does hold for
  actual `dev`/`start` runs connected to a real room (Twilio or otherwise)
  — just not for the console playground. No code fix applies here; once
  the upstream fix ships in a released livekit-agents version (see TODO
  above), this caveat goes away entirely regardless of testing mode.

## Full call transcript persistence (shipped 2026-07-16)
- `calls` and `call_turns` tables (`migrate_add_calls_tables.py`) store
  one row per call plus the ordered turn-by-turn transcript.
- `db.py` provides `create_call()` (opens the call record, returns its
  ID), `add_call_turn()` (persists one turn), and `finish_call()` (marks
  the call complete with its final result).
- `call_transcript.py`'s `CallTranscriptRecorder` queues conversation
  turns off the SDK's `conversation_item_added` event and writes them via
  a background worker task, so persistence never blocks call audio.
- Wired into both `agent.py` (`agent.py:1661/1718`) and `agent_v2.py`
  (`agent_v2.py:1423/1458`) — every real call now has a full transcript
  for auditing/compliance, not just the summarized outcome fields
  (`sales_status`, `recontact_at`, `referral_details`,
  `callback_details`, `objection_text`).

## OpenerTask hello-loop: VAD echo false positive (fixed 2026-07-14)
- Bug: `on_enter`'s hello-escalation loop (agent.py `OpenerTask.on_enter`)
  could silently exit after attempt 0 — no attempt 2/3, no timeout hangup,
  no logged error. Root cause: the `user_state_changed` listener was armed
  before `generate_reply` played Aiva's own "Hello?" TTS, so VAD echo/bleed
  from her own audio could flip state to "speaking" and set `heard_speech`
  before the loop ever reached its `wait_for` listen window — a false
  positive treated as "they responded," causing the coroutine to return
  early with no real conversation ever started.
- Fix applied: listener is now armed/disarmed per-attempt, scoped only to
  the actual `wait_for` window (after `generate_reply` returns, before the
  timeout race), with `heard_speech.clear()` beforehand to discard any
  echo captured during playback.
- Observed in Playground/console testing despite AEC ("aec warmup
  active/expired" in logs) — AEC did not fully suppress this. Possibly a
  browser-playground-specific limitation, similar to the webrtc-sys
  console-mode caveat above; real Twilio call audio path/echo
  characteristics may differ.
- KNOWN LIMITATION, not yet built: if this false-positive recurs on a real
  Twilio call, the next step is requiring an actual completed STT
  transcript as the "heard_speech" trigger instead of raw VAD
  speech-onset (`ev.new_state == "speaking"`). Not built now — revisit
  only if it resurfaces outside the playground.

## Same-breath/repeated-violation escalation + global safety nets (2026-07-17)
- Bug: a blocked `*_result` tool call (see `_require_confirmed_turn()` in
  `agent_v2.py`) could repeat indefinitely with no escalation, spinning a
  real call toward unbounded length. Fixed with a 3-strike per-stage
  escalation ladder (standard re-prompt -> forceful "don't repeat/don't
  guess" instruction -> defensive call-end), reusing the existing
  `_end_call()` chokepoint. Entirely generic (plain counter + escalating
  English instructions) — no model-specific logic, verified against
  Grok-4.1-fast-non-reasoning but not tuned to it.
- Testing then surfaced a DIFFERENT failure mode the ladder didn't cover:
  a model that never attempts any `*_result` tool call at all, just keeps
  generating text turn after turn (sometimes verbatim-repeating), plus a
  loophole where hopping between stages (e.g. opener -> disclosure ->
  opener) resets the per-stage violation counter before it ever reaches 3,
  letting a call evade the ladder indefinitely. Root cause of the
  no-tool-call stalling itself: no stage's instructions tell the model
  what to do when the prospect's response doesn't fit any of that stage's
  enumerated branches (e.g. a skeptical demand like "give me the name of
  who accesses my data" that isn't a yes/no to the pending script
  question) — it just answers the aside and never returns to the pending
  question or calls a tool.
- Fixed (this pass) with two additional counters, both plain/model-agnostic:
  1. `GLOBAL_TURN_LIMIT` (default 25) — total real user turns for the
     whole call, tracked independently of tool-call activity in
     `_on_conversation_item_added()`. Fires even if no tool is ever
     attempted.
  2. `WHOLE_CALL_VIOLATION_LIMIT` (default 5) — a whole-call violation
     counter alongside the existing per-stage one, that does NOT reset on
     `_set_stage()` — only on a genuine confirmed-turn success. Closes the
     stage-hopping loophole.
- KNOWN CONTENT GAP, not fixed in this pass: the underlying script gap
  (no stage has a universal "acknowledge the off-script question, defer
  what you don't know, then return to your pending question" instruction)
  is still open. The two counters above guarantee no call can run forever
  regardless, but a future pass should add that instruction across all
  `_STAGE_BLOCK_BUILDERS` blocks in `agent_v2.py` so the model has an
  actual valid branch instead of relying on the safety nets to end a
  stuck call defensively.

## Full-buffer same-breath guard confirmed as permanent design (2026-07-20)
- Full response buffering (not tail-only) is a deliberate, validated design
  decision, not a temporary workaround. Researched alternatives:
  1. Tail-only/sliding-window buffering — proven logically insufficient,
     since narration can precede a tool call by more than one sentence and
     there's no way to know the full shape of the response before it
     completes.
  2. API-level prevention via `tool_choice` constraints — confirmed
     unavailable across OpenAI, xAI/Grok, and Gemini APIs. `tool_choice`
     only controls whether/which tool is called, not whether text
     accompanies it; the one setting that does guarantee no bundled text
     (`tool_choice="required"`) forces a tool call on every turn and is
     incompatible with normal conversational replies. Gemini's `ANY` mode
     is documented to still leak text before the function call in some
     cases. Sources: OpenAI tool_choice="required" forum thread
     (community.openai.com/t/new-api-feature-forcing-function-calling-via-tool-choice-required/731488),
     xAI Function Calling guide (docs.x.ai/docs/guides/function-calling),
     LiveKit Cloud Inference Gateway docs (deepwiki.com/livekit/agents/5.7-livekit-cloud-inference-gateway),
     Gemini ANY-mode text leak report (discuss.ai.google.dev/t/when-using-function-calling-gemini-api-sends-a-text-response-along-with-function-call/84503).
- Measured cost: ~130-215ms added latency per substantive turn, negligible
  on short utterances. This cost is expected to persist even after
  switching to a stronger/paid model later, since the underlying model
  behavior (narrating alongside tool calls) is universal across providers,
  not specific to Grok.

## No-tool drift guard added to agent_v2.py (2026-07-20)
- Root cause confirmed for the same live call above: `current_stage` never
  left `"opener"` because the model carried a plausible-sounding
  pitch/qualifying/booking-style conversation entirely in free text,
  without ever attempting `opener_result`. Stage-tool scoping only
  restricts which tools the model CAN call — it does nothing to stop free
  text, so none of the real qualifying script, funding-estimate math, or
  booking script ever activated even though the conversation sounded
  coherent. This is distinct from the existing confirmed-turn violation
  ladder, which only fires when a tool call IS attempted and rejected.
- Fixed with a new, independent counter: `_no_tool_turn_streak`, tracking
  consecutive assistant turns within the same stage where the generation
  contained no tool call at all (`_last_generation_had_tool_call`, stamped
  once per generation at the tail of `llm_node()`, reusing the
  `tool_call_seen` value already computed there for the same-breath guard).
  Resets to 0 on any tool call and on every `_set_stage()` (a new stage is
  a fresh decision-point window). At `NO_TOOL_DRIFT_LIMIT` (3) consecutive
  no-tool turns, the guard arms a stage-aware corrective instruction for the
  next real user turn. It tells the model to resolve the current stage when
  the new answer is sufficient, or ask the specific missing question without
  guessing. It never starts another assistant generation itself. Wording is
  generic and stage-aware via `_STAGE_DECISION_POINTS`, not opener-specific.
  Orthogonal to the violation ladder by construction: a
  given turn either has a tool call attempt or doesn't, so the two
  mechanisms can never fire on the same turn.
- KNOWN, DELIBERATE GAP: `agent.py`'s `AgentTask` architecture has NO
  protection against a task drifting indefinitely without ever calling
  `task_complete` — no turn-counting infrastructure exists there at all
  (no `_user_turn_count`, no violation counters, nothing analogous to this
  guard). This is deliberate, not an oversight — deprioritized because
  `agent_v2.py` has become the primary focus of today's hardening work.
  Revisit if `agent.py` is ever brought back as a serious candidate.
