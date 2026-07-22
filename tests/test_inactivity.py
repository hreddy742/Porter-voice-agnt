import asyncio
import unittest
from unittest.mock import patch

from agent import (
    INACTIVITY_INSTRUCTION,
    INACTIVITY_REMINDERS,
    install_inactivity_handler,
)

REAL_SLEEP = asyncio.sleep


class Event:
    def __init__(self, new_state):
        self.new_state = new_state


class Speech:
    def __init__(self):
        self.played = False

    async def wait_for_playout(self):
        self.played = True


class Session:
    def __init__(self):
        self.handlers = {}
        self.replies = []

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)

    def emit(self, event, value):
        for callback in self.handlers[event]:
            callback(value)

    def generate_reply(self, **kwargs):
        speech = Speech()
        self.replies.append((kwargs, speech))
        return speech


class Agent:
    def __init__(self):
        self.is_ending = False
        self.end_calls = 0
        self.ended = asyncio.Event()

    async def end_for_inactivity(self):
        self.end_calls += 1
        self.is_ending = True
        self.ended.set()


class InactivityTest(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_retries_close_once(self):
        session = Session()
        agent = Agent()
        install_inactivity_handler(session, agent, "room-one")

        with patch("agent.asyncio.sleep", return_value=None):
            session.emit("user_state_changed", Event("away"))
            await asyncio.wait_for(agent.ended.wait(), timeout=1)

        self.assertEqual(len(session.replies), INACTIVITY_REMINDERS)
        self.assertEqual(agent.end_calls, 1)
        for options, speech in session.replies:
            self.assertEqual(options["instructions"], INACTIVITY_INSTRUCTION)
            self.assertTrue(options["allow_interruptions"])
            self.assertTrue(speech.played)

    async def test_duplicate_away_does_not_start_another_task(self):
        session = Session()
        agent = Agent()
        blocker = asyncio.Event()

        async def wait(_):
            await blocker.wait()

        install_inactivity_handler(session, agent, "room-one")
        with patch("agent.asyncio.sleep", side_effect=wait):
            session.emit("user_state_changed", Event("away"))
            session.emit("user_state_changed", Event("away"))
            await REAL_SLEEP(0)
            self.assertEqual(len(session.replies), 1)
            session.emit("user_state_changed", Event("listening"))
            await REAL_SLEEP(0)

        self.assertEqual(agent.end_calls, 0)

    async def test_user_speaking_cancels_reminders(self):
        session = Session()
        agent = Agent()
        blocker = asyncio.Event()

        async def wait(_):
            await blocker.wait()

        install_inactivity_handler(session, agent, "room-one")
        with patch("agent.asyncio.sleep", side_effect=wait):
            session.emit("user_state_changed", Event("away"))
            await REAL_SLEEP(0)
            session.emit("user_state_changed", Event("speaking"))
            await REAL_SLEEP(0)

        self.assertEqual(len(session.replies), 1)
        self.assertEqual(agent.end_calls, 0)

    async def test_ending_call_ignores_away_event(self):
        session = Session()
        agent = Agent()
        agent.is_ending = True
        install_inactivity_handler(session, agent, "room-one")
        session.emit("user_state_changed", Event("away"))
        await REAL_SLEEP(0)
        self.assertEqual(session.replies, [])

    async def test_session_close_cancels_reminders(self):
        session = Session()
        agent = Agent()
        blocker = asyncio.Event()

        async def wait(_):
            await blocker.wait()

        install_inactivity_handler(session, agent, "room-one")
        with patch("agent.asyncio.sleep", side_effect=wait):
            session.emit("user_state_changed", Event("away"))
            await REAL_SLEEP(0)
            session.emit("close", object())
            await REAL_SLEEP(0)

        self.assertEqual(agent.end_calls, 0)


if __name__ == "__main__":
    unittest.main()
