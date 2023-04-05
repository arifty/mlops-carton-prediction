"""
Generic Config class
"""
from shapeshifter.config.env_config import EnvConfig


class Config:
    """Configurations used in multiple other objects"""

    def __init__(self) -> None:
        self.env_config = EnvConfig()
        self.env = self.env_config.get("env")
        self.model_type = "lightgbm"
        self.project_name = self.env_config.get("project_name")
        self.s3_project_path = self.env_config.get("s3_keys.main")
        self.role = self.env_config.get("sagemaker_role")
        self.deployment_role = self.env_config.get("deployment_role")
        self.s3_bucket = "s3://" + self.env_config.get("s3_bucket")
        self.wi_s3_bucket = "s3://" + self.env_config.get("wi_s3_bucket")
        self.tags = [
            dict(self.env_config.get("sagemaker.tags.team")),
            dict(self.env_config.get("sagemaker.tags.product")),
        ]
        self.model_type = "lightgbm"
        self.root = "/opt/ml/processing/"
        self.scripts_root_path = f"{self.project_name}/steps/code/scripts"
