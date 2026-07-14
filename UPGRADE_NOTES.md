# Upgrade Tracker

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
  {"enable_thinking": False}}` (`app/agent/providers.py`
  `_build_llm("qwen")`), which drops warm-model latency to ~5s.
- Known remaining latency risk: Ollama unloads an idle model from memory
  after its default keep_alive window. A "cold" first call (model not
  currently loaded) measured ~24s before the tool call — still much
  better than the ~45s thinking-mode hang, but still a noticeable pause
  on a live call. Mitigated in part by `prewarm_qwen` in
  `app/agent/worker.py` when `LLM_FAILOVER_PROVIDER=qwen`.

## livekit-rtc webrtc-sys panic fix
- Bug: github.com/livekit/rust-sdks/issues/944
- Fix merged: rust-sdks PR #1098 (2026-07-09)
- Current livekit-agents (1.6.5) still pins livekit==1.1.13,
  which predates the fix
- Workaround applied: job_executor_type=JobExecutorType.PROCESS
  in `app/agent/worker.py` `run_worker()` (isolates crash to single
  call, not whole worker)
- TODO: once a livekit-agents release bundles rust-sdks with the
  fix (check pip index versions livekit for anything past 1.1.13
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
  agent changes. The PROCESS isolation is real and does hold for
  actual `dev`/`start` runs connected to a real room (Twilio or otherwise)
  — just not for the console playground. No code fix applies here; once
  the upstream fix ships in a released livekit-agents version (see TODO
  above), this caveat goes away entirely regardless of testing mode.

## OpenerTask hello-loop: VAD echo false positive (fixed 2026-07-14)
- Bug: `on_enter`'s hello-escalation loop
  (`app/agent/tasks/opener.py` `OpenerTask.on_enter`)
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
