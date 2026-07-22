import unittest
from unittest.mock import patch

from agent import build_turn_handling
from app.config import Settings
from livekit.plugins import deepgram


class TurnHandlingTest(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            agent_name="test-agent",
            agent_version="test",
            livekit_url="wss://livekit.internal.example.com",
            livekit_inference_api_key="inference-key",
            livekit_inference_api_secret="inference-secret",
            test_mode=True,
        )

    def test_exact_livekit_turn_configuration(self):
        detector = object()
        with patch("agent.inference.TurnDetector", return_value=detector) as factory:
            options = build_turn_handling(self.settings)

        factory.assert_called_once_with(
            version="v1",
            api_key="inference-key",
            api_secret="inference-secret",
        )
        self.assertIs(options["turn_detection"], detector)
        self.assertEqual(
            options["endpointing"],
            {"mode": "fixed", "min_delay": 0.3, "max_delay": 2.5},
        )
        self.assertEqual(
            options["interruption"],
            {
                "mode": "adaptive",
                "min_duration": 0.5,
                "min_words": 0,
                "false_interruption_timeout": 2.0,
                "resume_false_interruption": True,
            },
        )
        self.assertEqual(
            options["preemptive_generation"],
            {"enabled": True, "preemptive_tts": False},
        )

    def test_deepgram_supports_word_aligned_transcripts(self):
        stt = deepgram.STT(model="nova-2", api_key="test-key")
        self.assertEqual(stt.capabilities.aligned_transcript, "word")


if __name__ == "__main__":
    unittest.main()
