"""
Generic Sagemaker pipeline
"""
import json
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import boto3
from aws_utils.s3_lib import S3Proxy
from sagemaker.inputs import TransformInput, BatchDataCaptureConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import FrameworkProcessor
from sagemaker.estimator import Framework
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.transformer import Transformer
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import TransformStep, CacheConfig

from shapeshifter.config.env_config import EnvConfig
from shapeshifter.config.logging_config import LOGGING_CONFIG
from shapeshifter.utils import generic_functions

RARE_FEATURES = ["SIZE_CODE", "SILHOUETTE"]


class SagemakerPipeline(ABC):
    """
    Class containing common Sagemaker pipeline functionality.
    """

    def __init__(
        self,
        s3_lib: S3Proxy = None,
        rare_features: List[str] = RARE_FEATURES,
    ) -> None:
        """
        Init class
        """
        self.logger = generic_functions.get_logger(
            logging_dict=LOGGING_CONFIG, logger_name=__name__
        )
        self.env_config = EnvConfig()
        self.env = self.env_config.get("env")
        self.model_type = "lightgbm"
        self.s3_project_path = self.env_config.get("s3_keys.main")
        self.project_name = self.env_config.get("project_name")

        self.tags = [
            {"Key": "team", "Value": self.env_config.get("team_name")},
            {"Key": "product", "Value": self.project_name},
        ]

        self.sm_client = boto3.client("sagemaker", region_name="eu-west-1")
        self.s3_bucket = "s3://" + self.env_config.get("s3_bucket")
        self.s3_main_path = self.env_config.get("s3_keys.main")
        self.role = self.env_config.get("sagemaker_role")
        self.deployment_role = self.env_config.get("deployment_role")
        self.pipeline_session = PipelineSession(
            default_bucket=self.env_config.get("s3_bucket")
        )
        self.cache_config = CacheConfig(enable_caching=True, expire_after="1d")

        self.s3_paths = self.get_s3_paths()
        self.s3_data_drift_monitoring = self.env_config.get(
            "s3_keys.data_drift_monitoring"
        )

        self.s3_lib = (
            S3Proxy(
                bucket=self.env_config.get("s3_bucket"),
                endpoint_url=self.env_config.get("aws.s3_endpoint"),
            )
            if s3_lib is None
            else s3_lib
        )

        self.logger.info(
            f"role being used is: {self.pipeline_session.get_caller_identity_arn()}"
        )
        self.data_type = "text/csv"
        self.type: str = NotImplemented
        self.pipeline_name: str = NotImplemented
        self.pipeline_description: str = NotImplemented
        self.output_path: str = NotImplemented
        self.root = "/opt/ml/processing/"

        self.rare_features = rare_features
        self.rare_features_string = " ".join(self.rare_features)

    @property
    def training_config(self):
        return generic_functions.read_json_from_s3(
            s3_proxy=self.s3_lib,
            s3_bucket=self.env_config.get("s3_bucket"),
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
        self.logger.info("Shapeshifter inference ML pipeline run started.")
        self.trigger_sagemaker_pipeline()

    @abstractmethod
    def get_pipeline_parameters(self):
        raise NotImplementedError

    def trigger_sagemaker_pipeline(self):
        """Function for triggering a Sagemaker pipeline"""
        pipeline = Pipeline(
            name=self.pipeline_name,
        )

        execution = pipeline.start(
            parameters=self.get_pipeline_parameters(),
            execution_description=f"{self.type} pipeline triggered from Airflow",
        )
        self.logger.info(f"{self.type} pipeline triggered.")

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
        self.logger.info(f"role used for publishing pipeline: {self.role}")
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
        self.logger.info("Generate pipeline definition")
        self.logger.info(self.get_sagemaker_pipeline().definition())
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
                self.s3_bucket,
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

    def get_pyspark_processor(self) -> PySparkProcessor:
        return PySparkProcessor(
            base_job_name="knnights-spark",
            framework_version="3.1",
            role=self.role,
            instance_count=2,
            instance_type=self.env_config.get("sagemaker.processing_instance_type"),
            max_runtime_in_seconds=1200,
            sagemaker_session=self.pipeline_session,
            tags=self.tags,
        )

    def get_sklearn_processor(self) -> SKLearnProcessor:
        """Construct a sklearn processor to be used in steps in the pipeline"""
        return SKLearnProcessor(
            framework_version="1.0-1",
            role=self.role,
            instance_type=self.env_config.get("sagemaker.processing_instance_type"),
            instance_count=1,
            sagemaker_session=self.pipeline_session,
            tags=self.tags,
        )

    def get_batch_transform_step(
        self,
        model_name: str,
        data_uri: Union[str, Join],
        generate_inference_id: bool,
        input_filter: Optional[str] = None,
    ) -> TransformStep:
        """
        Constructs a TransformStep with Transformer instance

        Args:
            model_name (str): Name of the SageMaker model being used for the transform job.
            inputs (TransformInput): A sagemaker.inputs.TransformInput instance.

        Returns:
            (TransformStep): TransformStep for SageMaker Pipelines Workflows
        """
        transformer = Transformer(
            model_name=model_name,
            instance_type=self.env_config.get("sagemaker.inference_instance_type"),
            instance_count=1,
            output_path=self.output_path,
            sagemaker_session=self.pipeline_session,
            assemble_with="Line",
            accept=self.data_type,
            strategy="MultiRecord",
            max_payload=2,
            tags=self.tags,
        )

        step_transform = TransformStep(
            name="Transform",
            transformer=transformer,
            inputs=TransformInput(
                data=data_uri,
                content_type=self.data_type,
                split_type="Line",
                input_filter=input_filter,
                batch_data_capture_config=BatchDataCaptureConfig(
                    destination_s3_uri=Join(
                        values=[
                            self.s3_bucket,
                            self.env_config.get("s3_keys.data_capture_root"),
                        ],
                        on="/",
                    ),
                    generate_inference_id=generate_inference_id,
                ),
            ),
        )

        return step_transform

    @abstractmethod
    def get_quality_check_step(
        self,
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
    ) -> QualityCheckStep:
        """
        Function to create quality check from configs.

        Args:
            check_job_config (CheckJobConfig): job config
            quality_check_config (DataQualityCheckConfig): quality check config

        Returns:
            (QualityCheckStep): quality check step
        """
        raise NotImplementedError

    def get_baseline_data_drift_step(
        self,
        baseline_dataset: Union[str, PipelineVariable],
        output_s3_uri: Union[str, PipelineVariable],
    ) -> QualityCheckStep:
        """

        Args:
            baseline_dataset (Union[str, PipelineVariable]): baseline data set to use for data drift
            output_s3_uri (Union[str, PipelineVariable]): output s3 uri for the step

        Returns:
            (QualityCheckStep): step for data quality check
        """
        check_job_config = CheckJobConfig(
            role=self.role,
            instance_count=4,
            instance_type="ml.m5.4xlarge",
            volume_size_in_gb=120,
            sagemaker_session=self.pipeline_session,
            tags=self.tags,
        )

        data_quality_check_config = DataQualityCheckConfig(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_s3_uri,
        )

        data_quality_check_step = self.get_quality_check_step(
            check_job_config=check_job_config,
            quality_check_config=data_quality_check_config,
        )

        return data_quality_check_step

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
