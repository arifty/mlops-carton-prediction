from sagemaker.pytorch.estimator import PyTorch
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.functions import Join

from shapeshifter.config.logging_config import LOGGING_CONFIG
from shapeshifter.generic.config import Config
from shapeshifter.utils import generic_functions


class Executors(Config):
    """Executors for running jobs in Sagemaker"""

    def __init__(self, session) -> None:
        super().__init__()
        self.logger = generic_functions.get_logger(
            logging_dict=LOGGING_CONFIG, logger_name=__name__
        )
        self.session = session

        self.tuning_instance_type = self.env_config.get(
            "sagemaker.tuning_instance_type"
        )
        self.inference_instance_type = self.env_config.get(
            "sagemaker.inference_instance_type"
        )

    def get_sklearn_processor(self) -> SKLearnProcessor:
        """Construct a sklearn processor to be used in steps in the pipeline"""
        return SKLearnProcessor(
            framework_version="1.0-1",
            role=self.role,
            instance_type=self.env_config.get("sagemaker.processing_instance_type"),
            instance_count=1,
            sagemaker_session=self.session,
            tags=self.tags,
        )

    def get_tuner(
        self,
        objective_metric_name: str,
        hp_ranges: dict,
        max_jobs: int,
        max_parallel_jobs: int,
    ) -> HyperparameterTuner:
        """Construct a hyperparametertuner object for tuning a PyTorch model"""
        lightgbm_estimator = PyTorch(
            image_uri=self.env_config.get("sagemaker.docker_image_training"),
            source_dir=self.env_config.get("sagemaker.training_source_uri"),
            model_uri=self.env_config.get("sagemaker.prebuild_model_uri"),
            entry_point="transfer_learning.py",
            role=self.role,
            instance_count=1,
            instance_type=self.tuning_instance_type,
            output_path=Join(
                values=[
                    self.s3_bucket,
                    self.env_config.get("s3_keys.model_root"),
                    self.model_type,
                ],
                on="/",
            ),
            base_job_name=f"{self.project_name}-{self.model_type}",
            sagemaker_session=self.session,
        )

        metric_definitions = [
            {
                "Name": objective_metric_name,
                "Regex": f"{objective_metric_name}: ([0-9\.]+)",
            },
        ]

        # TODO: separate tuner from the step function
        tuner = HyperparameterTuner(
            estimator=lightgbm_estimator,
            objective_metric_name=objective_metric_name,
            objective_type="Minimize",
            metric_definitions=metric_definitions,
            hyperparameter_ranges=hp_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            tags=self.tags,
        )

        return tuner
