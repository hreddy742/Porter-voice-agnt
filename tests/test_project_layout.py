import ast
import unittest
from pathlib import Path

from agent import parse_console_lead_args, select_lead_for_session


ROOT = Path(__file__).parents[1]


class ProjectLayoutTest(unittest.TestCase):
    def test_console_lead_is_configurable_from_cli_arguments(self):
        livekit_args, lead = parse_console_lead_args(
            [
                "agent.py",
                "console",
                "--company-name",
                "Acme Manufacturing",
                "--contact-name",
                "Jordan",
                "--phone",
                "+15555550101",
                "--current-score",
                "88.5",
                "--text",
                "--log-level",
                "debug",
            ]
        )
        self.assertEqual(
            livekit_args,
            ["agent.py", "console", "--text", "--log-level", "debug"],
        )
        self.assertEqual(lead["company_name"], "Acme Manufacturing")
        self.assertEqual(lead["contact_name"], "Jordan")
        self.assertEqual(lead["phone"], "+15555550101")
        self.assertEqual(lead["current_score"], 88.5)

    def test_non_console_arguments_are_untouched(self):
        argv = ["agent.py", "dev", "--log-level", "debug"]
        self.assertEqual(parse_console_lead_args(argv), (argv, None))

    def test_livekit_only_console_arguments_do_not_override_database_lead(self):
        argv = ["agent.py", "console", "--text"]
        self.assertEqual(parse_console_lead_args(argv), (argv, None))

    def test_console_uses_non_persistent_lead_when_database_is_empty(self):
        lead, persist = select_lead_for_session(None, is_console=True)
        self.assertIsNotNone(lead)
        self.assertIsNone(lead["phone"])
        self.assertFalse(persist)

    def test_explicit_console_lead_is_used_without_persistence(self):
        expected = {"company_name": "Configured Co", "phone": None}
        lead, persist = select_lead_for_session(
            {"company_name": "Database Co", "phone": "+15555550101"},
            is_console=True,
            console_lead=expected,
        )
        self.assertEqual(lead, expected)
        self.assertFalse(persist)

    def test_real_job_still_requires_database_lead(self):
        lead, persist = select_lead_for_session(None, is_console=False)
        self.assertIsNone(lead)
        self.assertFalse(persist)

    def test_database_lead_can_have_or_omit_phone(self):
        for phone in ("+15555550101", None):
            expected = {"lead_candidate_id": "lead", "phone": phone}
            lead, persist = select_lead_for_session(expected, is_console=True)
            self.assertEqual(lead, expected)
            self.assertTrue(persist)

    def test_canonical_entrypoint_uses_agent_server(self):
        source = (ROOT / "agent.py").read_text(encoding="utf-8")
        self.assertIn("AgentServer", source)
        self.assertIn("@server.rtc_session", source)
        self.assertIn("cli.run_app(server)", source)
        self.assertIn('sys.argv[1:2] != ["console"]', source)
        self.assertIn("is_console = ctx.is_fake_job()", source)
        self.assertIn("allow_livekit_cloud=is_console", source)

    def test_controller_has_no_model_node_overrides(self):
        source = (ROOT / "app/agent/voice_agent.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        methods = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({"llm_node", "tts_node", "transcription_node"}.isdisjoint(methods))

    def test_removed_modules_stay_removed(self):
        for path in (
            "app/agent/providers.py",
            "app/agent/boundary_audio.py",
            "app/agent/clips.py",
            "app/agent/observability.py",
            "app/infrastructure/transcript.py",
            "app/domain/knowledge.py",
        ):
            self.assertFalse((ROOT / path).exists())

    def test_entrypoint_uses_only_fixed_voice_providers(self):
        source = (ROOT / "agent.py").read_text(encoding="utf-8")
        for expected in (
            'deepgram.STT(model="nova-2")',
            'openai.LLM(model="gpt-4o-mini")',
            "cartesia.TTS(",
            'voice="a33f7a4c-100f-41cf-a1fd-5822e8fc253f"',
            'proc.userdata["vad"] = inference.VAD(',
        ):
            self.assertIn(expected, source)
        self.assertNotIn("app.agent.providers", source)
        self.assertNotIn("livekit.plugins import silero", source)

    def test_entrypoint_uses_livekit_away_detection(self):
        source = (ROOT / "agent.py").read_text(encoding="utf-8")
        self.assertIn("user_away_timeout=15.0", source)
        self.assertIn('session.on("user_state_changed"', source)
        self.assertIn("UserStateChangedEvent", source)

    def test_entrypoint_uses_livekit_turn_handling(self):
        source = (ROOT / "agent.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        session_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AgentSession"
        )
        session_keywords = {keyword.arg for keyword in session_call.keywords}

        self.assertIn("turn_handling", session_keywords)
        self.assertNotIn("preemptive_generation", session_keywords)
        for expected in (
            "TurnHandlingOptions(",
            'version="v1"',
            '"mode": "fixed"',
            '"min_delay": 0.3',
            '"max_delay": 2.5',
            '"mode": "adaptive"',
            '"false_interruption_timeout": 2.0',
            '"resume_false_interruption": True',
            '"preemptive_tts": False',
        ):
            self.assertIn(expected, source)

    def test_agent_uses_prebuilt_livekit_end_call_tool(self):
        source = (ROOT / "app/agent/voice_agent.py").read_text(encoding="utf-8")
        self.assertIn("from livekit.agents.beta.tools import EndCallTool", source)
        self.assertIn("EndCallTool(", source)
        self.assertNotIn("async def end_call(", source)


if __name__ == "__main__":
    unittest.main()
