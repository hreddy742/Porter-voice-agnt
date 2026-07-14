"""PorterSupervisor — routes the call through AgentTasks."""

from livekit.agents import Agent, JobContext, function_tool

from app.agent.speech import SpeechSafetyMixin
from app.agent.tasks import (
    BookingTask,
    DisclosureTask,
    ExitTask,
    ObjectionTask,
    OpenerTask,
    PitchTask,
    QualifierTask,
)


# ============================================================
# SUPERVISOR
# ============================================================

class PorterSupervisor(SpeechSafetyMixin, Agent):

    def __init__(self, lead, ctx: JobContext):
        self.lead        = lead
        self.ctx         = ctx
        self.call_result = 'no_answer'

        company  = lead.get('company_name', 'the company')
        city     = lead.get('city', '')
        state    = lead.get('state', '')
        industry = lead.get('industry', 'staffing')
        tier     = lead.get('tier', 'warm')

        super().__init__(
            instructions=f"""
            Porter Capital sales call. Agent: Aiva (AI).
            Lead: {company}, {city} {state}, {industry}, Tier: {tier}

            ABSOLUTE RULE: NEVER quote rates or numbers. Ever.
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
                instructions=closing_instructions,
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
            # Not a sales decision — a routing dead end. There's no DB
            # column for referral details today, so they're logged to
            # console/call notes only; call_result falls back to
            # 'contacted' (real contact was made, just not with a decision
            # maker) rather than inventing a new status literal db.py's
            # update_lead_status() doesn't accept.
            referral_details = result.get('referral_details', '')
            print(f"[Gatekeeper referral] {self.lead.get('company_name', 'lead')}: {referral_details}")
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
        if result == 'qualified':
            return "Qualified. Move toward booking."
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


