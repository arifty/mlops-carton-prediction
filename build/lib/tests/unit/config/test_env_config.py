import unittest

from shapeshifter.config.env_config import EnvConfig


class TestEnvConfig(unittest.TestCase):
    def test_init(self):
        env_config = EnvConfig()
        self.assertEqual(env_config.env, "local")

        with self.assertRaises(SystemExit):
            EnvConfig(fallback_env="not_valid_env")

    def test_get(self):
        env_config = EnvConfig()

        self.assertEqual(env_config.get("env"), "local")
        self.assertEqual(env_config.get("team_name"), "knnights")

        with self.assertRaises(KeyError):
            env_config.get("no_key_defined")
