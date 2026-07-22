import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ProjectLayoutTest(unittest.TestCase):
    def test_entrypoint_and_command_modules_are_discoverable(self):
        self.assertTrue((ROOT / "agent_v2.py").is_file())
        self.assertIsNotNone(importlib.util.find_spec("scripts.call"))
        self.assertIsNotNone(importlib.util.find_spec("scripts.simulate"))
        self.assertIsNotNone(
            importlib.util.find_spec("tests.manual.test_opener_cleanpath_live_manual")
        )


if __name__ == "__main__":
    unittest.main()
