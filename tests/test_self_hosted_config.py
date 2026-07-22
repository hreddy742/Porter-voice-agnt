import os
import unittest
from unittest.mock import patch

import agent
from app.config import Settings


class SelfHostedConfigTest(unittest.TestCase):
    inference_env = {
        "LIVEKIT_INFERENCE_API_KEY": "inference-key",
        "LIVEKIT_INFERENCE_API_SECRET": "inference-secret",
    }

    def test_cloud_hostname_is_rejected(self):
        with patch.dict(
            os.environ,
            {"LIVEKIT_URL": "wss://example.livekit.cloud", **self.inference_env},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "self-hosted"):
                Settings.from_env()

        with patch.dict(
            os.environ,
            {"LIVEKIT_URL": "wss://livekit.cloud", **self.inference_env},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "self-hosted"):
                Settings.from_env()

    def test_self_hosted_hostname_is_accepted(self):
        url = "wss://livekit.internal.example.com"
        with patch.dict(
            os.environ,
            {"LIVEKIT_URL": url, **self.inference_env},
            clear=True,
        ):
            self.assertEqual(Settings.from_env().livekit_url, url)

    def test_cloud_hostname_is_allowed_only_for_console_jobs(self):
        url = "wss://example.livekit.cloud"
        with patch.dict(
            os.environ,
            {"LIVEKIT_URL": url, **self.inference_env},
            clear=True,
        ):
            settings = Settings.from_env(allow_livekit_cloud=True)
            self.assertEqual(settings.livekit_url, url)

    def test_inference_credentials_are_required(self):
        with patch.dict(
            os.environ,
            {"LIVEKIT_URL": "wss://livekit.internal.example.com"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Inference credentials"):
                Settings.from_env()

    def test_livekit_credentials_are_inference_fallback(self):
        with patch.dict(
            os.environ,
            {
                "LIVEKIT_URL": "wss://livekit.internal.example.com",
                "LIVEKIT_API_KEY": "livekit-key",
                "LIVEKIT_API_SECRET": "livekit-secret",
            },
            clear=True,
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.livekit_inference_api_key, "livekit-key")
            self.assertEqual(settings.livekit_inference_api_secret, "livekit-secret")

    def test_dedicated_inference_credentials_take_precedence(self):
        with patch.dict(
            os.environ,
            {
                "LIVEKIT_URL": "wss://livekit.internal.example.com",
                "LIVEKIT_API_KEY": "self-hosted-key",
                "LIVEKIT_API_SECRET": "self-hosted-secret",
                **self.inference_env,
            },
            clear=True,
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.livekit_inference_api_key, "inference-key")
            self.assertEqual(settings.livekit_inference_api_secret, "inference-secret")

    def test_console_skips_only_the_worker_preflight(self):
        with (
            patch.object(agent.sys, "argv", ["agent.py", "console"]),
            patch.object(agent.Settings, "from_env") as from_env,
            patch.object(agent.cli, "run_app") as run_app,
        ):
            agent.main()
        from_env.assert_not_called()
        run_app.assert_called_once_with(agent.server)

    def test_dev_keeps_the_self_hosted_preflight(self):
        with (
            patch.object(agent.sys, "argv", ["agent.py", "dev"]),
            patch.object(agent.Settings, "from_env") as from_env,
            patch.object(agent.cli, "run_app") as run_app,
        ):
            agent.main()
        from_env.assert_called_once_with()
        run_app.assert_called_once_with(agent.server)


if __name__ == "__main__":
    unittest.main()
