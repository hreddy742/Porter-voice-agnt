import unittest
from unittest.mock import patch

from livekit.agents.llm import ChatMessage

from agent import log_transcript


class Event:
    def __init__(self, item):
        self.item = item


class Agent:
    current_stage = "qualifying"


class TranscriptLoggingTest(unittest.TestCase):
    def test_logs_role_stage_room_and_text(self):
        event = Event(ChatMessage(role="user", content=["We invoice businesses."]))
        with patch("agent.logger.info") as info:
            log_transcript(event, "room-one", Agent())
        args = info.call_args.args
        self.assertIn("transcript", args[0])
        self.assertEqual(args[1:], ("room-one", "user", "qualifying", "We invoice businesses."))


if __name__ == "__main__":
    unittest.main()
