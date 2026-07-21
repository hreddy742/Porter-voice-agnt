# ponytail: minimal self-check for the two-range rate guardrail
# (find_rate_violation in agent.py / agent_v2.py). Run: python test_rate_guardrail.py

import agent
import agent_v2


def _check(mod):
    # in-range fee mention, hedged, "to" separator -> clean
    assert mod.find_rate_violation(
        "Our rate typically runs somewhere between 0.5 to 3 percent, but we'd build you an actual number."
    ) is None

    # in-range fee mention, hedged, "and" separator (the canonical locked
    # script phrasing, e.g. RATE_FALLBACK_LINE itself) -> clean
    assert mod.find_rate_violation(
        "Rates typically run somewhere between 0.5 and 3 percent — "
        "I'll get you the exact number through one of our advisors."
    ) is None

    # in-range advance mention, "to" separator -> clean
    assert mod.find_rate_violation(
        "We typically advance somewhere between 80 to 95 percent of the invoice value upfront."
    ) is None

    # in-range advance mention, "and" separator (the canonical locked line) -> clean
    assert mod.find_rate_violation(
        "We typically advance somewhere between 80 and 95 percent of the "
        "invoice value upfront — the exact number depends on your customers "
        "and how the deal is structured."
    ) is None

    # out-of-range fee range -> flagged, fee fallback
    v = mod.find_rate_violation("Our rate is 0.2 to 2 percent depending on volume.")
    assert v is not None and v[1] == mod.RATE_FALLBACK_LINE

    # out-of-range advance range -> flagged, advance fallback
    v = mod.find_rate_violation("We advance 10 to 20 percent of the invoice value upfront.")
    assert v is not None and v[1] == mod.ADVANCE_FALLBACK_LINE

    # in-range advance number, but no context cue at all -> conservative flag
    v = mod.find_rate_violation("The number is 90 percent, just so you know.")
    assert v is not None and v[1] == mod.RATE_FALLBACK_LINE

    # bare unhedged fee number, even in range -> flagged (only the range is sanctioned)
    v = mod.find_rate_violation("Our fee is 1 percent.")
    assert v is not None

    print(f"{mod.__name__}: all rate-guardrail checks passed")


if __name__ == "__main__":
    _check(agent)
    _check(agent_v2)
