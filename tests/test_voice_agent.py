import unittest

from app.agent.voice_agent import Aiva
from app.agent.prompt import build_context_variables
from livekit.agents.llm import ToolContext


LEAD = {
    "lead_candidate_id": "00000000-0000-0000-0000-000000000001",
    "company_name": "Example Co",
    "website_domain": "example.com",
}


class FakeSpeech:
    def __init__(self):
        self.played = False

    async def wait_for_playout(self):
        self.played = True


class FakeSession:
    def __init__(self):
        self.replies = []
        self.shutdown_calls = []

    def generate_reply(self, **kwargs):
        speech = FakeSpeech()
        self.replies.append((kwargs, speech))
        return speech

    def shutdown(self, *, drain):
        self.shutdown_calls.append(drain)


class FakeActivity:
    def __init__(self, session):
        self.session = session
        self.tools = []

    async def update_tools(self, tools):
        self.tools = tools


class FakeContext:
    def __init__(self):
        self.deleted = 0

    async def delete_room(self):
        self.deleted += 1


class FakeRepository:
    def __init__(self):
        self.suppressions = []

    async def add_to_suppression(self, company_name, website_domain, reason="opted_out"):
        self.suppressions.append((company_name, website_domain, reason))


def tool_names(agent):
    return {
        tool.info.name if hasattr(tool, "info") else tool.id
        for tool in agent._current_tools()
    }


async def invoke(agent, name, *args, **kwargs):
    tool = getattr(agent, name)
    return await tool._func(agent, *args, **kwargs)


class VoiceAgentTest(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, *, test_mode=False):
        ctx = FakeContext()
        repository = FakeRepository()
        agent = Aiva(
            LEAD,
            build_context_variables(LEAD),
            ctx,
            repository,
            test_mode=test_mode,
        )
        session = FakeSession()
        agent._activity = FakeActivity(session)
        return agent, session, ctx, repository

    async def test_livekit_initial_tool_context_has_unique_function_names(self):
        agent, _, _, _ = self.make_agent()
        flattened = ToolContext(agent.tools).flatten()
        names = [tool.info.name for tool in flattened]
        self.assertEqual(len(names), len(set(names)))

    async def test_greeting_uses_generate_reply(self):
        agent, session, _, _ = self.make_agent()
        await agent.on_enter()
        self.assertEqual(session.replies[0][0]["instructions"], 'Begin the call now. Say only "Hello?"')

    async def test_prompt_context_is_separate_from_application_lead(self):
        agent, _, _, _ = self.make_agent()
        self.assertEqual(agent.context_variables["company"], "Example Co")
        self.assertEqual(agent.lead["website_domain"], "example.com")
        self.assertNotIn("website_domain", agent.context_variables)

    async def test_happy_path_preserves_stage_order(self):
        agent, _, _, _ = self.make_agent()
        await invoke(agent, "opener_result", "interested")
        self.assertEqual(agent.current_stage, "pitch")
        await invoke(agent, "pitch_result", "interested")
        self.assertEqual(agent.current_stage, "qualifying")
        await invoke(agent, "qualifying_result", "qualified")
        self.assertEqual(agent.current_stage, "booking")

    async def test_objection_resumes_interrupted_stage(self):
        agent, _, _, _ = self.make_agent()
        await invoke(agent, "opener_result", "interested")
        await invoke(agent, "pitch_result", "objection", "already funded")
        self.assertEqual(agent.current_stage, "objection")
        await invoke(agent, "objection_result", "still_interested")
        self.assertEqual(agent.current_stage, "pitch")
        self.assertEqual(agent.objection_log, ["already funded"])

    async def test_opener_terminal_results_keep_business_outcomes(self):
        cases = {
            "not_interested": "not_interested",
            "bad_timing": "callback_later",
            "wrong_number": "not_interested",
            "hung_up": "no_answer",
            "gatekeeper_referral": "contacted",
        }
        for result, expected in cases.items():
            with self.subTest(result=result):
                agent, _, ctx, _ = self.make_agent()
                await invoke(agent, "opener_result", result, "Taylor, extension twelve")
                self.assertEqual(agent.call_result, expected)
                self.assertEqual(ctx.deleted, 1)
                if result == "gatekeeper_referral":
                    self.assertEqual(agent.referral_details, "Taylor, extension twelve")

    async def test_qualification_and_disclosure_branches(self):
        agent, _, _, _ = self.make_agent()
        agent.current_stage = "qualifying"
        await invoke(agent, "qualifying_result", "not_qualified")
        self.assertEqual(agent.call_result, "not_interested")

        agent, _, _, _ = self.make_agent()
        agent.current_stage = "pitch"
        await invoke(agent, "enter_disclosure")
        self.assertEqual(agent.current_stage, "disclosure")
        await invoke(agent, "disclosure_result", "accepted")
        self.assertEqual(agent.current_stage, "pitch")

        agent, _, _, _ = self.make_agent()
        agent.current_stage = "disclosure"
        await invoke(agent, "disclosure_result", "wants_human")
        self.assertEqual(agent.current_stage, "booking")

    async def test_objection_terminal_and_callback_branches(self):
        expected = {
            "not_interested": ("not_interested", "objection"),
            "bad_timing": ("callback_later", "objection"),
            "wants_callback": ("no_answer", "booking"),
        }
        for result, (call_result, stage) in expected.items():
            with self.subTest(result=result):
                agent, _, _, _ = self.make_agent()
                agent.current_stage = "objection"
                await invoke(agent, "objection_result", result)
                self.assertEqual(agent.call_result, call_result)
                self.assertEqual(agent.current_stage, stage)

    async def test_booking_is_available_from_every_nonterminal_stage(self):
        agent, _, _, _ = self.make_agent()
        for stage in ("opener", "pitch", "qualifying", "objection", "disclosure"):
            agent.current_stage = stage
            self.assertIn("booking_result", tool_names(agent))

    async def test_livekit_end_call_tool_is_stage_scoped(self):
        agent, _, _, _ = self.make_agent()
        for stage in ("opener", "pitch", "qualifying", "objection", "booking", "disclosure"):
            agent.current_stage = stage
            self.assertIn("end_call", tool_names(agent))
        agent.current_stage = "exit"
        self.assertNotIn("end_call", tool_names(agent))

    async def test_end_call_tool_marks_user_ended_outcome(self):
        agent, _, _, _ = self.make_agent()
        await agent._on_end_call_tool_called(None)
        self.assertTrue(agent.is_ending)
        self.assertEqual(agent.call_result, "not_interested")

        agent, _, _, _ = self.make_agent()
        agent.call_result = "contacted"
        await agent._on_end_call_tool_called(None)
        self.assertEqual(agent.call_result, "contacted")

    async def test_booking_closes_after_playout(self):
        agent, session, ctx, _ = self.make_agent()
        await invoke(agent, "booking_result", "Friday at ten on 555 0100")
        self.assertEqual(agent.call_result, "callback_booked")
        self.assertTrue(session.replies[-1][1].played)
        self.assertEqual(session.shutdown_calls, [True])
        self.assertEqual(ctx.deleted, 1)

    async def test_inactivity_uses_no_answer_closing(self):
        agent, session, ctx, _ = self.make_agent()
        await agent.end_for_inactivity()
        self.assertEqual(agent.call_result, "no_answer")
        self.assertIn("inactivity closing message", session.replies[-1][0]["instructions"])
        self.assertTrue(session.replies[-1][1].played)
        self.assertEqual(session.shutdown_calls, [True])
        self.assertEqual(ctx.deleted, 1)

    async def test_opt_out_is_persisted_before_closing(self):
        agent, _, ctx, repository = self.make_agent()
        await invoke(agent, "enter_exit")
        self.assertEqual(repository.suppressions, [("Example Co", "example.com", "opted_out")])
        self.assertEqual(agent.call_result, "suppressed")
        self.assertEqual(ctx.deleted, 1)

    async def test_test_mode_skips_suppression_write(self):
        agent, _, _, repository = self.make_agent(test_mode=True)
        await invoke(agent, "enter_exit")
        self.assertEqual(repository.suppressions, [])


if __name__ == "__main__":
    unittest.main()
