from typing import List, Optional, Tuple, Union

from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, IntegerParameter
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import ProcessingStep, Step, TuningStep

from shapeshifter.steps.generic_steps import SagemakerSteps


class TrainingSteps(SagemakerSteps):
    """Steps used for the training pipeline"""

    def __init__(self, session, output_path: str) -> None:
        super().__init__(session, output_path)

    def get_input_processor_step(
        self,
        input_data: Union[str, PipelineVariable],
        output_path: Union[str, PipelineVariable],
    ) -> ProcessingStep:
        """Define the input processor"""

        # input_data = self.training_params.get("s3_train_input")

        step_process = ProcessingStep(
            name="PrepData",
            processor=self.get_sklearn_processor(),
            code=f"{self.scripts_root_path}/train_input_processing.py",
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination=f"{self.root}input",
                ),
                ProcessingInput(
                    source=f"{self.project_name}/data/modeling_schema.json",
                    destination=f"{self.root}input/schema",
                    input_name="modeling_schema.json",
                    s3_input_mode="File",
                ),
            ],
            job_arguments=[
                "--input-s3-filename",
                input_data,
                "--rare-features",
                f"{self.rare_features_string}",
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train_no_target_with_header",
                    source=f"{self.root}processed/train_no_target_with_header",
                    destination=Join(
                        values=[
                            output_path,
                            "train_no_target_with_header",
                        ],
                        on="/",
                    ),
                ),
                ProcessingOutput(
                    output_name=self.model_type,
                    source=f"{self.root}processed/{self.model_type}",
                    destination=Join(
                        values=[output_path, "lightgbm"],
                        on="/",
                    ),
                ),
                ProcessingOutput(
                    output_name="encoders",
                    source=f"{self.root}processed/encoders",
                    destination=Join(
                        values=[output_path, "encoders"],
                        on="/",
                    ),
                ),
            ],
            cache_config=self.cache_config,
        )
        return step_process

    def get_tune_model_step(
        self, train_data_s3_path: str, max_jobs: int, max_parallel_jobs: int
    ) -> TuningStep:
        """Define the tuning object for tuning the model

        Args:
            train_data_s3_path (str): s3 of the training dataset

        Returns:
            TuningStep: Definition of the tuning step
        """
        objective_metric_name = "rmse"

        # TODO: move out to a more controllable place
        hyperparameter_ranges = {
            "learning_rate": ContinuousParameter(1e-4, 0.1, scaling_type="Logarithmic"),
            "num_boost_round": IntegerParameter(10, 500),
            "num_leaves": IntegerParameter(10, 100),
            "feature_fraction": ContinuousParameter(0.1, 1),
            "bagging_fraction": ContinuousParameter(0.1, 1),
            "bagging_freq": IntegerParameter(1, 10),
            "max_depth": IntegerParameter(5, 30),
            "min_data_in_leaf": IntegerParameter(10, 500),
            "tweedie_variance_power": ContinuousParameter(1, 1.99),
            "boosting": CategoricalParameter(["gbdt", "dart"]),
            "max_bin": IntegerParameter(2, 100),
            "lambda_l1": ContinuousParameter(0, 100),
            "lambda_l2": ContinuousParameter(0, 100),
            # "min_gain_to_split": ContinuousParameter(0, 15),
        }

        tuner = self.get_tuner(
            objective_metric_name=objective_metric_name,
            hp_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
        )

        step_lightgbm_tuning = TuningStep(
            name="HPTuning",
            step_args=tuner.fit(inputs=TrainingInput(s3_data=train_data_s3_path)),
        )

        return step_lightgbm_tuning

    def get_quality_check_step(
        self,
        baseline_data_statitics: Optional[Union[str, PipelineVariable]],
        baseline_data_constraints: Optional[Union[str, PipelineVariable]],
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
        register_new_baseline: Union[bool, PipelineVariable] = False,
    ) -> QualityCheckStep:
        data_quality_check_step = QualityCheckStep(
            name="BaselineDataCreationStep",
            skip_check=True,
            register_new_baseline=register_new_baseline,
            quality_check_config=quality_check_config,
            check_job_config=check_job_config,
            model_package_group_name=f"{self.project_name}-{self.model_type}",
            cache_config=self.cache_config,
            display_name="data_baseline_creation",
            description="Baseline creation for data drift checks",
        )

        return data_quality_check_step

    def get_evaluate_model_step(
        self,
        actuals_data_s3_uri: str,
        inference_data_s3_uri: str,
        evaluation_script_path: str,
    ) -> Tuple[ProcessingStep, PropertyFile]:
        """Use the created model and a test dataset to evaluate the model.
           In the provided script, you can define what metrics to calculate.

        Args:
            data_s3_uri (str): s3 path to the data file to be used
            evaluation_script_path (str): path to the script used for the evaluation

        Returns:
            Tuple[ProcessingStep, PropertyFile]: return the evaluate step and the propertyfile
        """
        evaluation_report = PropertyFile(
            name="EvaluationReport", output_name="evaluation", path="evaluation.json"
        )

        step_evaluate_lightgbm = ProcessingStep(
            name="EvaluateModelLightGBM",
            processor=self.get_sklearn_processor(),
            code=evaluation_script_path,
            inputs=[
                ProcessingInput(
                    source=actuals_data_s3_uri,
                    destination=f"{self.root}input",
                ),
                ProcessingInput(
                    source=inference_data_s3_uri,
                    destination=f"{self.root}input/{self.model_type}",
                ),
                ProcessingInput(
                    source=f"{self.project_name}/data/modeling_schema.json",
                    destination=f"{self.root}input/schema",
                    input_name="modeling_schema.json",
                    s3_input_mode="File",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source=f"{self.root}evaluation",
                    destination=Join(
                        values=[inference_data_s3_uri, "evaluation"],
                        on="/",
                    ),
                ),
            ],
            property_files=[evaluation_report],
        )

        return step_evaluate_lightgbm, evaluation_report

    def get_register_model_step(
        self,
        model: PyTorchModel,
        model_metrics: ModelMetrics,
        drift_check_baselines: DriftCheckBaselines,
        dependencies: List[Step],
    ) -> Tuple[ModelStep, ModelStep]:
        # Create a RegisterModel step, which registers the model with Sagemaker Model Registry.
        model_registry_params = {
            "content_types": ["text/csv"],
            "response_types": ["text/csv"],
            "inference_instances": ["ml.m5.large"],
            "transform_instances": ["ml.m5.large"],
            "model_package_group_name": f"{self.project_name}-{self.env_config.get('env')}-{self.model_type}",
            "model_metrics": model_metrics,
            "drift_check_baselines": drift_check_baselines,
            "description": "Model for Shapeshifter predictions based on LightGBM",
            "image_uri": self.env_config.get("sagemaker.docker_image_inference"),
        }

        step_register_lightgbm_model = ModelStep(
            name="Register",
            step_args=model.register(
                approval_status="PendingManualApproval", **model_registry_params
            ),
            depends_on=dependencies,
        )

        step_register_lightgbm_model_approval = ModelStep(
            name="RegisterApproved",
            step_args=model.register(
                approval_status="Approved", **model_registry_params
            ),
            depends_on=dependencies,
        )

        return step_register_lightgbm_model, step_register_lightgbm_model_approval

    def get_config_step(self, job_arguments: List) -> ProcessingStep:
        step_config = ProcessingStep(
            name="SaveConfig",
            processor=self.get_sklearn_processor(),
            code=f"{self.scripts_root_path}/create_config.py",
            outputs=[
                ProcessingOutput(
                    source=f"{self.root}config",
                    destination=Join(
                        values=[
                            self.wi_s3_bucket,
                            self.env_config.get("s3_keys.config_root"),
                        ],
                        on="/",
                    ),
                ),
            ],
            job_arguments=job_arguments,
            cache_config=self.cache_config,
        )
        return step_config

    def get_condition_step(
        self,
        evaluation_step: ProcessingStep,
        evaluation_report: PropertyFile,
        if_steps: List[Step],
        else_steps: List[Step],
        mae_threshold: float,
    ) -> ConditionStep:
        # Create accuracy condition to ensure the model meets performance requirements.
        # Models with a test accuracy lower than the condition will not be registered with the model registry.
        cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="regression_metrics.mean_absolute_error.value",
            ),
            right=mae_threshold,
        )

        # Create a Sagemaker Pipelines ConditionStep, using the condition above.
        # Enter the steps to perform if the condition returns True / False.
        step_cond = ConditionStep(
            name="MAELowerThanThresholdCondition",
            conditions=[cond_lte],
            if_steps=if_steps,
            else_steps=else_steps,
        )

        return step_cond
