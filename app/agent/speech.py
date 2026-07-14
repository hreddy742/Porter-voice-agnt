"""Natural speech guide and TTS leak filtering for Aiva."""

import re

from livekit.agents import Agent


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


