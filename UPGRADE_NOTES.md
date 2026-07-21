# Upgrade Tracker

## Deterministic stage progression: forced tool_choice + stamp rollback + qualifying's "unclear" value (2026-07-21)
Closes out the no-tool-drift root-cause fix (previously `NO_TOOL_DRIFT_LIMIT`
was a backstop nudge only). `_inject_drift_correction()` (agent_v2.py) now:

- Forces the stage's own `*_result` tool via `tool_choice={"type":
  "function", "function": {"name": tool_name}}` on the corrective
  `generate_reply()` call, instead of just asking nicely. Confirmed
  supported by both OpenAI and Groq (our primary/failover pair) —
  `livekit.plugins.groq.LLM` is a thin subclass of the OpenAI plugin
  pointed at Groq's OpenAI-compatible endpoint, same code path for both.
- Rolls `_last_agent_speech_turn_count` back to `_user_turn_count - 1`
  before firing the forced call. Without this, `_require_confirmed_turn()`
  (called inside every `*_result` tool) silently discarded the forced
  call's effect — it saw "no user turn since the question was last asked"
  because that stamp gets re-set after every assistant turn, including the
  drifted no-tool ones, even though a real user answer had just arrived.
  The rollback fixes the specific case (real answer already given, never
  acted on) without weakening the guard's actual purpose (never let the
  model answer its own brand-new question with zero real user input).
- `qualifying_result` gained a fourth `result` value, `unclear`, plus
  instruction text telling the model to base arguments strictly on what
  was actually said and use `unclear` rather than guess a specific outcome
  when it genuinely doesn't have enough information yet.

**Validated live against real gpt-4o-mini**, not just unit-tested plumbing:
reproduced the exact drift scenario (3 no-tool turns in `qualifying`, then
forced resolution) via `test_drift_correction_live_manual.py`. Confirmed:
`current_stage` now actually advances after the forced call (previously
stuck), and a vague/insufficient-info conversation correctly produced
`qualifying_result(result="unclear", ...)` with a genuine follow-up
question instead of a fabricated `not_qualified` guess.

Known limits, explicitly not resolved by this change:
- **Sample size**: validated on one clean run per scenario, not a large
  sample. LLM output isn't perfectly deterministic — occasional bad
  guesses in Scenario-B-style (vague/insufficient-info) cases are reduced,
  not proven eliminated. Re-run `test_drift_correction_live_manual.py` if
  a similar bad-guess incident shows up in production transcripts.
- **Only `qualifying_result` has the `unclear` escape hatch.**
  `opener_result`, `pitch_result`, `objection_result`, `disclosure_result`,
  and `booking_result` do NOT — if drift correction ever forces one of
  those tools, it carries the same forced-too-early bad-guess risk
  demonstrated above, currently unaddressed. See the comment above
  `_STAGE_INSUFFICIENT_INFO_VALUE` in agent_v2.py.
- `NO_TOOL_DRIFT_LIMIT`, the whole-call violation counter
  (`_total_violation_count`), and `GLOBAL_TURN_LIMIT` are all kept
  unchanged as backstops, not replaced.

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
  no-tool turns, `_inject_drift_correction()` forces a strong corrective
  instruction telling the model to stop general conversation and either
  resolve the current stage's decision point now (naming its actual
  `*_result` tool) or ask the specific question needed to do so — wording
  is generic and stage-aware via `_STAGE_DECISION_POINTS`, not
  opener-specific. Orthogonal to the violation ladder by construction: a
  given turn either has a tool call attempt or doesn't, so the two
  mechanisms can never fire on the same turn.
- KNOWN, DELIBERATE GAP: `agent.py`'s `AgentTask` architecture has NO
  protection against a task drifting indefinitely without ever calling
  `task_complete` — no turn-counting infrastructure exists there at all
  (no `_user_turn_count`, no violation counters, nothing analogous to this
  guard). This is deliberate, not an oversight — deprioritized because
  `agent_v2.py` has become the primary focus of today's hardening work.
  Revisit if `agent.py` is ever brought back as a serious candidate.
