""" Environment configuration class"""
import logging
import os
import sys
from collections import OrderedDict
from functools import lru_cache

from pyhocon import ConfigFactory
from pyhocon.config_tree import ConfigTree
from pyhocon.exceptions import ConfigMissingException


@lru_cache(maxsize=32)
class EnvConfig:
    """
    Class to retrieve configurations from hocon config file.
    """

    allowed_envs = ["local", "test", "qa", "prod"]

    def __init__(
        self,
        fallback_env="local",
        config_dict=os.path.join(os.path.dirname(__file__), "env_config.conf"),
    ):
        """
        Init method

        Args:
            fallback_env (str): which environment to run in when no env is found in hocon file.
            config_dict (str): which .conf file to read, by default reads config.conf file within same folder,
                                default to config_file=os.path.join(os.path.dirname(__file__), "env_config.conf"
        """
        self.logger = logging.getLogger(__name__)
        param_odict = OrderedDict()
        param_odict["generic.env"] = fallback_env
        param_config = ConfigFactory.from_dict(param_odict)
        self.conf = ConfigFactory.parse_file(config_dict, resolve=False).with_fallback(
            param_config
        )
        self.env = self.conf.get_string("generic.env")
        self.logger.info(f"using environment '{self.env}'")

        if self.env not in self.allowed_envs:
            sys.exit(
                f"{self.env} is not one of the allowed environments {self.allowed_envs}"
            )

        self.conf = ConfigTree.merge_configs(
            self.conf.get("generic"), self.conf.get(f"env_specific.{self.env}")
        )

    @lru_cache(maxsize=32)
    def get(self, key: str):
        """
        Function to get environment config variable from key.

        Args:
            key (str): key in config

        Returns:
            environment config value for given key

        """
        try:
            value = self.conf.get(key)
            self.logger.info(f"using value '{value}' for config key '{key}'")
            return value
        except ConfigMissingException as err:
            raise KeyError(f"{key} not found in config file,") from err


if __name__ == "__main__":
    print(EnvConfig().get("sagemaker.training.accuracy_mae_threshold"))
