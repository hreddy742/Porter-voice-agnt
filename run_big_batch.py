"""
============================================================
100-scenario batch: expands simulate.py's 7 categories (14 hand-written
scenarios) with persona-LLM-generated variations to reach ~14-15 per
category, runs all of them through the real TextAiva harness, grades every
transcript, and produces an aggregate BATCH SUMMARY covering pass/fail,
which safety net (if any) ended each call, rate-guardrail activations,
leaked tool-call syntax, new failure patterns, and the "never calls a tool"
stalling-loop recurrence rate.

Purely additive — does not modify simulate.py, agent_v2.py, or agent.py.
Reuses simulate.py's SCENARIOS/run_scenario/grade_transcript/_ollama_chat
unmodified; the only "instrumentation" is a non-behavior-changing wrapper
around av.strip_leaked_meta_text to silently count how often a raw leak
was actually caught (simulate.py always looks this up as an av.* attribute
at call time, so the wrapper is picked up transparently).
============================================================
"""

import asyncio
import contextlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import simulate as sim
import agent_v2 as av

TARGET_COUNTS = {
    "compound_questions": 14,
    "skeptical_honesty": 14,
    "emotionally_charged": 14,
    "rate_baiting": 14,
    "off_script": 14,
    "interruptions": 15,
    "ambiguous": 15,
}
ID_PREFIX = {
    "compound_questions": "compound",
    "skeptical_honesty": "skeptical",
    "emotionally_charged": "emotional",
    "rate_baiting": "rate_bait",
    "off_script": "off_script",
    "interruptions": "interrupt",
    "ambiguous": "ambiguous",
}

CATEGORY_THEME = {
    "compound_questions": "opens with several questions stacked into one breath, rushed or transactional tone",
    "skeptical_honesty": "distrustful, probes hard on data privacy, contracts, or whether the agent is being straight with them",
    "emotionally_charged": "irritated, burned before, or emotionally reactive to a cold call",
    "rate_baiting": "fixated on getting a hard rate/percentage number immediately, pushy about it",
    "off_script": "asks a genuine but unusual operational question outside the normal script, calm/curious tone",
    "interruptions": "conversation flow gets disrupted — flip-flopping interest, asking to repeat, trailing off, cutting in",
    "ambiguous": "vague, non-committal, hard-to-classify answers throughout ('maybe', 'I guess', 'sure, whatever')",
}


# ============================================================
# Leak-catch instrumentation — transparent wrapper, same output
# ============================================================
_leak_catch_count = 0
_orig_strip_leaked_meta_text = av.strip_leaked_meta_text


def _instrumented_strip(text):
    global _leak_catch_count
    if av._FUNCTION_LEAK.search(text) or av._FUNCTION_LEAK_UNCLOSED.search(text):
        _leak_catch_count += 1
    return _orig_strip_leaked_meta_text(text)


av.strip_leaked_meta_text = _instrumented_strip


# ============================================================
# Scenario generation — persona LLM fills out each category
# ============================================================

def _generate_variations(category, existing, n_needed):
    if n_needed <= 0:
        return []
    examples = "\n".join(
        f'- persona: "{s["persona"]}" / opening_line: "{s["opening_line"]}"'
        for s in existing
    )
    prompt = f"""Generate {n_needed} DISTINCT new prospect personas for a cold-call
sales-agent test suite, category theme: {CATEGORY_THEME[category]}.

Existing examples in this category (for tone/style reference only — do not
repeat or closely paraphrase these):
{examples}

Each persona needs:
- "persona": a short (1-2 sentence) description of the prospect's attitude/situation
- "opening_line": a single natural spoken line this prospect would say picking up a cold call, fitting the persona

Respond with ONLY a raw JSON array of exactly {n_needed} objects, each with
keys "persona" and "opening_line". No markdown, no commentary, no code fences."""

    for attempt in range(3):
        raw = sim._ollama_chat([{"role": "user", "content": prompt}], temperature=0.9)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list) and all(
                isinstance(d, dict) and "persona" in d and "opening_line" in d for d in data
            ):
                return data[:n_needed]
        except json.JSONDecodeError:
            pass
    # Fallback: never block the batch on a flaky generation — pad with
    # light variations of the existing examples rather than fail outright.
    print(f"  [WARN] generation failed for {category} after 3 attempts — using fallback padding")
    fallback = []
    for i in range(n_needed):
        base = existing[i % len(existing)]
        fallback.append({
            "persona": base["persona"] + f" (fallback variant {i + 1})",
            "opening_line": base["opening_line"],
        })
    return fallback


def build_100_scenarios():
    by_category = {}
    for s in sim.SCENARIOS:
        by_category.setdefault(s["category"], []).append(s)

    full = []
    for category, target in TARGET_COUNTS.items():
        existing = by_category.get(category, [])
        full.extend(existing)
        needed = target - len(existing)
        print(f"Generating {needed} variations for '{category}' (target {target})...")
        variations = _generate_variations(category, existing, needed)
        prefix = ID_PREFIX[category]
        next_n = len(existing) + 1
        for v in variations:
            full.append({
                "category": category,
                "id": f"{prefix}_{next_n}",
                "persona": v["persona"],
                "opening_line": v["opening_line"],
            })
            next_n += 1

    assert len(full) == 100, f"expected 100 scenarios, got {len(full)}"
    return full


# ============================================================
# Per-scenario run with captured console markers
# ============================================================

_STAGE_GUARD_RE = re.compile(r"\[VIOLATION GUARD\] repeated confirmation failure in stage '(\w+)'")
_WHOLE_CALL_RE = re.compile(r"\[GLOBAL VIOLATION LIMIT\]")
_TURN_LIMIT_RE = re.compile(r"\[GLOBAL TURN LIMIT\]")
_RATE_GUARDRAIL_RE = re.compile(r"\[RATE GUARDRAIL\] blocked spoken text \(([^)]+)\)")

_LEAK_CHECK_1 = re.compile(r"<function\b", re.IGNORECASE)
_LEAK_CHECK_2 = re.compile(r"\bcall\s+\w+_(result|out)\b", re.IGNORECASE)


def _detect_stalling_run(entries):
    """Longest run of consecutive assistant turns with NO tool call attempt
    (tool_call entries reset the run). >=3 flags the 'never calls a tool'
    stalling pattern for this transcript."""
    max_run = current = 0
    for e in entries:
        if e["speaker"] == "assistant" and not e.get("tool"):
            current += 1
            max_run = max(max_run, current)
        elif e["speaker"] == "tool_call":
            current = 0
    return max_run


async def run_one(scenario, run_dir, idx, total):
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        result = await sim.run_scenario(scenario, run_dir)
    elapsed = time.time() - t0
    captured = buf.getvalue()

    stage_guard = _STAGE_GUARD_RE.search(captured)
    whole_call = bool(_WHOLE_CALL_RE.search(captured))
    turn_limit = bool(_TURN_LIMIT_RE.search(captured))
    rate_hits = _RATE_GUARDRAIL_RE.findall(captured)

    entries = [json.loads(l) for l in Path(result["transcript_path"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    stall_run = _detect_stalling_run(entries)

    raw_leak_in_log = any(
        e["speaker"] == "assistant" and (_LEAK_CHECK_1.search(e["text"]) or _LEAK_CHECK_2.search(e["text"]))
        for e in entries
    )

    ending = "natural"
    if turn_limit:
        ending = "global_turn_limit"
    elif whole_call:
        ending = "whole_call_violation"
    elif stage_guard:
        ending = "per_stage_violation_guard"
    elif result["error"]:
        ending = "max_turns_or_crash"

    line = (
        f"[{idx}/{total}] {scenario['id']} ({scenario['category']}) — "
        f"{'PASS' if result['passed'] else 'FAIL'} ending={ending} "
        f"call_result={result['call_result']} turns={result['turns']} "
        f"({elapsed:.0f}s)"
    )
    print(line, flush=True)

    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "passed": result["passed"],
        "call_result": result["call_result"],
        "error": result["error"],
        "turns": result["turns"],
        "transcript_path": result["transcript_path"],
        "ending": ending,
        "stage_guard_stage": stage_guard.group(1) if stage_guard else None,
        "rate_guardrail_hits": rate_hits,
        "stalling_run_length": stall_run,
        "stalling_detected": stall_run >= 3,
        "raw_leak_in_log": raw_leak_in_log,
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================
# Batch summary
# ============================================================

def summarize(results, grades):
    total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = total - n_pass

    ending_counts = Counter(r["ending"] for r in results)
    ending_by_category = {}
    for r in results:
        ending_by_category.setdefault(r["category"], Counter())[r["ending"]] += 1

    stage_guard_stages = Counter(r["stage_guard_stage"] for r in results if r["stage_guard_stage"])

    rate_hits_total = sum(len(r["rate_guardrail_hits"]) for r in results)
    rate_hit_scenarios = [r["scenario_id"] for r in results if r["rate_guardrail_hits"]]

    raw_leaks_in_log = [r["scenario_id"] for r in results if r["raw_leak_in_log"]]

    stalling = [r for r in results if r["stalling_detected"]]
    stalling_by_category = Counter(r["category"] for r in stalling)

    fail_reasons = Counter(r["error"] for r in results if not r["passed"])

    high_sev_findings = [
        {"scenario_id": g["scenario_id"], **f}
        for g in grades for f in g["findings"] if f.get("severity") == "high"
    ]

    summary = {
        "total_runs": total,
        "passed": n_pass,
        "failed": n_fail,
        "fail_reasons": dict(fail_reasons),
        "ending_reason_counts": dict(ending_counts),
        "ending_reason_by_category": {k: dict(v) for k, v in ending_by_category.items()},
        "per_stage_violation_guard_stage_counts": dict(stage_guard_stages),
        "rate_guardrail_total_activations": rate_hits_total,
        "rate_guardrail_scenarios": rate_hit_scenarios,
        "raw_leak_caught_count": _leak_catch_count,
        "raw_leak_found_in_final_log": raw_leaks_in_log,
        "stalling_loop_detected_count": len(stalling),
        "stalling_loop_rate": round(len(stalling) / total, 3),
        "stalling_loop_by_category": dict(stalling_by_category),
        "grader_total_findings": sum(len(g["findings"]) for g in grades),
        "grader_high_severity_findings": high_sev_findings,
    }
    return summary


async def main():
    print("Building 100-scenario set (14 existing + persona-LLM-generated variations)...")
    scenarios = build_100_scenarios()
    run_dir = sim._run_dir()
    (run_dir / "scenarios_100.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    print(f"Scenario set ready: {len(scenarios)} total. Run dir: {run_dir}\n")

    results = []
    t_start = time.time()
    for i, scenario in enumerate(scenarios, 1):
        result = await run_one(scenario, run_dir, i, len(scenarios))
        results.append(result)
        if i % 10 == 0:
            elapsed_min = (time.time() - t_start) / 60
            print(f"--- progress checkpoint: {i}/100 done, {elapsed_min:.1f} min elapsed ---\n", flush=True)

    (run_dir / "batch_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nRunning grader pass over all 100 transcripts...")
    grades = []
    for r in results:
        g = sim.grade_transcript(r)
        grades.append(g)
    (run_dir / "grades.json").write_text(json.dumps(grades, indent=2), encoding="utf-8")

    summary = summarize(results, grades)
    (run_dir / "BATCH_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE — {total_min:.1f} minutes total, run dir: {run_dir}")
    print(json.dumps(summary, indent=2))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if sim._aiva_llm is not None:
            asyncio.run(sim._aiva_llm.aclose())
