import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class StaticRegressionTest(unittest.TestCase):
    def test_removed_runtime_patterns_do_not_return(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "app").rglob("*.py")
        )
        for forbidden in (
            "psycopg2",
            "CallTranscriptRecorder",
            "add_call_turn",
            "strip_leaked_meta_text",
            "find_rate_violation",
            "PORTER_CAPITAL_KNOWLEDGE",
            "LLM_FAILOVER_PROVIDER",
            "LLM_PROVIDER",
            "TTS_PROVIDER",
        ):
            self.assertNotIn(forbidden, source)

    def test_unused_provider_dependencies_are_removed(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("livekit-plugins-groq", requirements)
        self.assertNotIn("livekit-plugins-elevenlabs", requirements)

    def test_no_transcript_database_schema_or_write(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "app", ROOT / "scripts")
            for path in path.rglob("*.py")
        )
        self.assertNotIn("call_turns", source)


if __name__ == "__main__":
    unittest.main()
