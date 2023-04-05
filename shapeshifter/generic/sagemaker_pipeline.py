"""
Generic Sagemaker pipeline
"""
import json
from abc import ABCMeta, abstractmethod
from typing import Tuple
import logging

import boto3
from aws_utils.s3_lib import S3Proxy
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.session import Session

from shapeshifter.config.logging_config import LOGGING_CONFIG
from shapeshifter.generic.config import Config
from shapeshifter.utils import generic_functions


class SagemakerPipeline(Config, metaclass=ABCMeta):
    """
    Class containing common Sagemaker pipeline functionality.
    """

    def __init__(
        self,
        s3_lib: S3Proxy = None,
    ) -> None:
        """
        Init class
        """
        super().__init__()
        self.logger = generic_functions.get_logger(
            logging_dict=LOGGING_CONFIG, logger_name=__name__
        )

        self.sm_client = boto3.client("sagemaker", region_name="eu-west-1")
        self.s3_main_path = self.env_config.get("s3_keys.main")
        self.pipeline_session = PipelineSession(
            default_bucket=self.env_config.get("s3_bucket"),
        )

        self.s3_paths = self.get_s3_paths()
        self.s3_data_drift_monitoring = self.env_config.get(
            "s3_keys.data_drift_monitoring"
        )

        self.s3_lib = (
            S3Proxy(
                bucket=self.env_config.get("wi_s3_bucket"),
                endpoint_url=self.env_config.get("aws.s3_endpoint"),
            )
            if s3_lib is None
            else s3_lib
        )

        logging.info(
            f"role being used is: {self.pipeline_session.get_caller_identity_arn()}"
        )
        self.type: str = NotImplemented
        self.pipeline_name: str = NotImplemented
        self.pipeline_description: str = NotImplemented
        self.output_path: str = NotImplemented
        self.root = "/opt/ml/processing/"

    @property
    def training_config(self):
        return generic_functions.read_json_from_s3(
            s3_proxy=self.s3_lib,
            s3_bucket=self.env_config.get("wi_s3_bucket"),
            s3_path_key=self.env_config.get("s3_keys.train_config"),
        )

    def run(
        self,
        start_date: str,
        end_date: str,
    ):
        """
        Run different steps of ML pipelines

        Args:

        """
        logging.info("Shapeshifter inference ML pipeline run started.")
        self.trigger_sagemaker_pipeline()

    @abstractmethod
    def get_pipeline_parameters(self):
        raise NotImplementedError

    def trigger_sagemaker_pipeline(self):
        """Function for triggering a Sagemaker pipeline"""
        pipeline_parameters = self.get_pipeline_parameters()

        # Switch to CICD role for triggering the actual pipeline
        # generic_functions.get_assume_role_credentials(role_arn=self.deployment_role)
        sm_session = Session(
            boto_session=generic_functions.get_boto3_session(self.deployment_role),
            default_bucket=self.env_config.get("s3_bucket"),
        )
        logging.info(
            f"role being used for trigger is: {sm_session.get_caller_identity_arn()}"
        )
        pipeline = Pipeline(name=self.pipeline_name, sagemaker_session=sm_session)

        execution = pipeline.start(
            parameters=pipeline_parameters,
            execution_description=f"{self.type} pipeline triggered from Airflow",
        )
        logging.info(f"{self.type} pipeline triggered.")

        execution.wait(delay=60, max_attempts=120)

    @abstractmethod
    def get_sagemaker_pipeline(self) -> Pipeline:
        """Abstract getter for the pipeline"""
        raise NotImplementedError

    def publish_sagemaker_pipeline(self) -> dict:
        """
        Function to publish sagemaker pipeline

        Returns: reponse

        """
        logging.info(f"role used for publishing pipeline: {self.role}")
        response = self.get_sagemaker_pipeline().upsert(
            role_arn=self.role,
            description=self.pipeline_description,
            tags=self.tags,
        )
        return response

    def generate_pipeline_definition(self) -> dict:
        """From the sagemaker pipeline, generate the json like definition

        Returns:
            dict: the pipeline deifinition
        """
        logging.info("Generate pipeline definition")
        logging.info(self.get_sagemaker_pipeline().definition())
        return self.get_sagemaker_pipeline().definition()

    def get_s3_paths(self) -> dict:
        """Constructs and returns all s3 paths used in the sagemaker pipelines

        Returns:
            dict: set of s3 paths to be used in the rest of the pipelines
        """
        inference_test_path = Join(
            values=[
                self.s3_bucket,
                self.env_config.get("s3_keys.inference"),
                "test_predictions",
                ExecutionVariables.START_DATETIME,
            ],
            on="/",
        )

        inference_live_path = Join(
            values=[
                self.wi_s3_bucket,
                self.env_config.get("s3_keys.inference"),
                "live",
                ExecutionVariables.START_DATETIME,
            ],
            on="/",
        )

        data_processed_path = Join(
            values=[
                self.s3_bucket,
                self.env_config.get("s3_keys.data_root"),
                "processed",
                ExecutionVariables.START_DATETIME,
            ],
            on="/",
        )

        return {
            "inference_live_path": inference_live_path,
            "inference_test_path": inference_test_path,
            "data_processed_path": data_processed_path,
        }

    def parse_modeling_schema(self, file_path: str) -> Tuple[list, list, str, dict]:
        """From the schema json, get all the different components

        Args:
            file_path (str): path to the modeling schema file

        Returns:
            Tuple[list, list, str, dict]: returns a list of keys, features and the target variable.
                                        Next to this also a dictionary with the dtypes for those elements.
        """
        with open(file_path) as f:
            modeling_schema = json.load(f)

        keys = list(modeling_schema.get("keys").keys())
        features = list(modeling_schema.get("features").keys())
        target = list(modeling_schema.get("target").keys())[0]

        dtypes = {}
        for element in ["keys", "features", "target"]:
            dtypes.update(modeling_schema.get(element).items())

        return keys, features, target, dtypes
