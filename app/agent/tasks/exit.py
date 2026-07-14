"""Exit / opt-out task — acknowledge and suppress."""

from livekit.agents import AgentTask, function_tool

from app.agent.speech import NATURAL_SPEECH_GUIDE, SpeechSafetyMixin
from app.db import add_to_suppression


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


