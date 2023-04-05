import os
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

from sagemaker.inputs import BatchDataCaptureConfig, TransformInput
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.transformer import Transformer
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import CacheConfig, TransformStep

from shapeshifter.config.env_config import EnvConfig
from shapeshifter.config.logging_config import LOGGING_CONFIG
from shapeshifter.steps.executors import Executors
from shapeshifter.utils import generic_functions

RARE_FEATURES = ["SIZE_CODE", "SILHOUETTE"]


class SagemakerSteps(Executors, metaclass=ABCMeta):
    def __init__(
        self,
        session,
        output_path: str,
        rare_features: List[str] = RARE_FEATURES,
    ) -> None:
        super().__init__(session=session)
        self.logger = generic_functions.get_logger(
            logging_dict=LOGGING_CONFIG, logger_name=__name__
        )
        self.session = session
        self.output_path = output_path

        self.rare_features = rare_features
        self.rare_features_string = " ".join(self.rare_features)
        self.cache_config = CacheConfig(enable_caching=True, expire_after="1d")
        self.data_type = "text/csv"

    @abstractmethod
    def get_input_processor_step(self, input_data: str, output_path: str):
        """Abstract class for getting the input processor"""
        raise NotImplementedError

    def get_create_model_step(self, model: Join) -> Tuple[ModelStep, PyTorchModel]:
        """Create the model step

        Args:
            model (Join): s3 uri to the model object
        """
        best_lightgbm_model = PyTorchModel(
            name="LightGBM",
            image_uri=self.env_config.get("sagemaker.docker_image_inference"),
            model_data=model,
            source_dir=f"{self.project_name}/steps/code/inference_code",
            sagemaker_session=self.session,
            entry_point="inference.py",
            role=self.role,
            code_location=os.path.join(
                self.s3_bucket,
                self.env_config.get("s3_keys.main"),
                "models",
                "repacked_model",
            ),
        )

        step_create_best_lightgbm = ModelStep(
            name="LightGBM",
            step_args=best_lightgbm_model.create(
                instance_type=self.env_config.get("sagemaker.inference_instance_type"),
                tags=self.tags,
            ),
        )
        return step_create_best_lightgbm, best_lightgbm_model

    def get_batch_transform_step(
        self,
        model_name: str,
        data_uri: Union[str, Join],
        generate_inference_id: bool = False,
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
        # TODO: separate transformer from the step function
        transformer = Transformer(
            model_name=model_name,
            instance_type=self.env_config.get("sagemaker.inference_instance_type"),
            instance_count=1,
            output_path=self.output_path,
            sagemaker_session=self.session,
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
                input_filter=input_filter,
            ),
        )

        return step_transform

    @abstractmethod
    def get_quality_check_step(
        self,
        baseline_data_statitics: Optional[Union[str, PipelineVariable]],
        baseline_data_constraints: Optional[Union[str, PipelineVariable]],
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
        register_new_baseline: Union[bool, PipelineVariable] = False,
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
        baseline_data_statitics: Optional[Union[str, PipelineVariable]] = None,
        baseline_data_constraints: Optional[Union[str, PipelineVariable]] = None,
        register_new_baseline: Union[bool, PipelineVariable] = False,
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
            sagemaker_session=self.session,
            tags=self.tags,
        )

        data_quality_check_config = DataQualityCheckConfig(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_s3_uri,
        )

        data_quality_check_step = self.get_quality_check_step(
            baseline_data_statitics=baseline_data_statitics,
            baseline_data_constraints=baseline_data_constraints,
            check_job_config=check_job_config,
            quality_check_config=data_quality_check_config,
            register_new_baseline=register_new_baseline,
        )

        return data_quality_check_step
