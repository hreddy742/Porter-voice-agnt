"""Porter Capital conversation state and LiveKit function tools."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from livekit.agents import Agent, JobContext, function_tool
from livekit.agents.beta.tools import EndCallTool

from .prompt import ContextVariables, build_system_prompt


logger = logging.getLogger(__name__)


class SuppressionRepository(Protocol):
    async def add_to_suppression(
        self, company_name: str, website_domain: str, reason: str = "opted_out"
    ) -> None: ...


Stage = Literal[
    "opener", "pitch", "qualifying", "objection", "booking", "disclosure", "exit"
]

_STAGE_TOOLS: dict[Stage, tuple[str, ...]] = {
    "opener": ("opener_result",),
    "pitch": ("pitch_result",),
    "qualifying": ("qualifying_result",),
    "objection": ("objection_result",),
    "booking": ("booking_result",),
    "disclosure": ("disclosure_result",),
    "exit": (),
}


class Aiva(Agent):
    """A single continuous agent that owns the complete call workflow."""

    def __init__(
        self,
        lead: Mapping[str, Any],
        context_variables: ContextVariables,
        ctx: JobContext,
        repository: SuppressionRepository,
        *,
        test_mode: bool = False,
    ) -> None:
        self.lead = dict(lead)
        self.context_variables = dict(context_variables)
        self.ctx = ctx
        self.repository = repository
        self.test_mode = test_mode
        self.current_stage: Stage = "opener"
        self.call_result = "no_answer"
        self.referral_details: str | None = None
        self.callback_details: str | None = None
        self.objection_log: list[str] = []
        self._return_stage: Stage | None = None
        self._call_ending = False
        self._end_call_tool = EndCallTool(
            extra_description=(
                "Use only when the prospect explicitly asks to end or hang up. "
                "Do not use for opt-out requests or instead of a stage result tool."
            ),
            delete_room=True,
            end_instructions=(
                "Use the system prompt's user_ended closing message exactly."
            ),
            on_tool_called=self._on_end_call_tool_called,
        )
        super().__init__(
            instructions=build_system_prompt(context_variables),
        )

    def _current_tools(self, *, include_end_call: bool = True) -> list[Any]:
        names = list(_STAGE_TOOLS[self.current_stage])
        if self.current_stage not in {"booking", "exit"}:
            names.append("booking_result")
        if self.current_stage not in {"disclosure", "exit"}:
            names.append("enter_disclosure")
        if self.current_stage != "exit":
            names.append("enter_exit")
        tools = [getattr(self, name) for name in dict.fromkeys(names)]
        if include_end_call and self.current_stage != "exit":
            tools.append(self._end_call_tool)
        return tools

    async def _on_end_call_tool_called(self, _: Any) -> None:
        self._call_ending = True
        if self.call_result == "no_answer":
            self.call_result = "not_interested"
        logger.info(
            "end_call_tool_called stage=%s result=%s",
            self.current_stage,
            self.call_result,
        )

    async def _set_stage(self, stage: Stage) -> None:
        self.current_stage = stage
        await self.update_tools(self._current_tools())
        logger.info("stage_changed stage=%s", stage)

    def _continue(self) -> None:
        self.session.generate_reply()

    @property
    def is_ending(self) -> bool:
        return self._call_ending

    async def end_for_inactivity(self) -> None:
        if self._call_ending:
            return
        self.call_result = "no_answer"
        await self._close("inactivity")

    async def _close(self, outcome: str, *, speak: bool = True) -> None:
        if self._call_ending:
            return
        self._call_ending = True
        try:
            if speak:
                speech = self.session.generate_reply(
                    instructions=(
                        f"Close the call now using only the system prompt's "
                        f"{outcome} closing message."
                    ),
                    allow_interruptions=False,
                )
                await speech.wait_for_playout()
        finally:
            self.session.shutdown(drain=True)
            await self.ctx.delete_room()

    async def on_enter(self) -> None:
        await self.update_tools(self._current_tools(include_end_call=False))
        greeting = self.session.generate_reply(
            instructions='Begin the call now. Say only "Hello?"',
            allow_interruptions=True,
        )
        await greeting.wait_for_playout()
        if not self._call_ending:
            await self.update_tools(self._current_tools())

    @function_tool()
    async def opener_result(
        self,
        result: Literal[
            "interested",
            "not_interested",
            "bad_timing",
            "wrong_number",
            "hung_up",
            "gatekeeper_referral",
            "needs_clarification",
        ],
        referral_details: str = "",
    ) -> None:
        """Record the completed opener result and optional referral details."""
        if self._call_ending or self.current_stage != "opener":
            return
        if result == "needs_clarification":
            self._continue()
        elif result == "gatekeeper_referral":
            self.referral_details = referral_details or None
            self.call_result = "contacted"
            await self._close("gatekeeper_referral")
        elif result == "wrong_number":
            self.call_result = "not_interested"
            await self._close("wrong_number")
        elif result == "not_interested":
            self.call_result = "not_interested"
            await self._close("not_interested")
        elif result == "bad_timing":
            self.call_result = "callback_later"
            await self._close("bad_timing")
        elif result == "hung_up":
            self.call_result = "no_answer"
            await self._close("hung_up", speak=False)
        else:
            self.call_result = "contacted"
            await self._set_stage("pitch")
            self._continue()

    @function_tool()
    async def pitch_result(
        self,
        result: Literal["interested", "objection"],
        objection_text: str = "",
    ) -> None:
        """Record whether the pitch was accepted or raised an objection."""
        if self._call_ending or self.current_stage != "pitch":
            return
        if result == "objection":
            await self._raise_objection(objection_text)
        else:
            await self._set_stage("qualifying")
            self._continue()

    @function_tool()
    async def qualifying_result(
        self,
        result: Literal["qualified", "not_qualified", "objection", "unclear"],
        objection_text: str = "",
    ) -> None:
        """Record the completed qualification result and optional objection."""
        if self._call_ending or self.current_stage != "qualifying":
            return
        if result == "unclear":
            self._continue()
        elif result == "objection":
            await self._raise_objection(objection_text)
        elif result == "not_qualified":
            self.call_result = "not_interested"
            await self._close("not_qualified")
        else:
            await self._set_stage("booking")
            self._continue()

    async def _raise_objection(self, objection_text: str) -> None:
        self._return_stage = self.current_stage
        if objection_text:
            self.objection_log.append(objection_text)
        await self._set_stage("objection")
        self._continue()

    @function_tool()
    async def objection_result(
        self,
        result: Literal[
            "still_interested", "not_interested", "bad_timing", "wants_callback"
        ],
    ) -> None:
        """Record the completed objection-handling result."""
        if self._call_ending or self.current_stage != "objection":
            return
        if result == "not_interested":
            self.call_result = "not_interested"
            await self._close("not_interested")
        elif result == "bad_timing":
            self.call_result = "callback_later"
            await self._close("bad_timing")
        elif result == "wants_callback":
            await self._set_stage("booking")
            self._continue()
        else:
            await self._set_stage(self._return_stage or "qualifying")
            self._continue()

    @function_tool()
    async def booking_result(self, callback_details: str) -> None:
        """Record confirmed callback contact and timing details."""
        if self._call_ending:
            return
        await self._set_stage("booking")
        self.callback_details = callback_details
        self.call_result = "callback_booked"
        await self._close("booking")

    @function_tool()
    async def enter_disclosure(self) -> None:
        """Enter disclosure when the prospect asks whether the agent is human."""
        if self._call_ending or self.current_stage == "disclosure":
            return
        self._return_stage = self.current_stage
        await self._set_stage("disclosure")
        self._continue()

    @function_tool()
    async def disclosure_result(
        self, reaction: Literal["accepted", "wants_human", "wants_to_end"]
    ) -> None:
        """Record the prospect's response to the AI disclosure."""
        if self._call_ending or self.current_stage != "disclosure":
            return
        if reaction == "wants_human":
            await self._set_stage("booking")
            self._continue()
        elif reaction == "wants_to_end":
            self.call_result = "not_interested"
            await self._close("wants_to_end")
        else:
            await self._set_stage(self._return_stage or "opener")
            self._continue()

    @function_tool()
    async def enter_exit(self) -> None:
        """Apply a do-not-call request and end the conversation."""
        if self._call_ending:
            return
        await self._set_stage("exit")
        if self.test_mode:
            logger.info(
                "suppression_skipped test_mode=true company=%r domain=%r",
                self.lead.get("company_name", ""),
                self.lead.get("website_domain", ""),
            )
        else:
            await self.repository.add_to_suppression(
                str(self.lead.get("company_name") or ""),
                str(self.lead.get("website_domain") or ""),
            )
        self.call_result = "suppressed"
        await self._close("opt_out")
