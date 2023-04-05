"""
Training Sagemaker pipeline
"""
import os

from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.quality_check_step import QualityCheckStep

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline
from shapeshifter.steps.training_steps import TrainingSteps


class TrainSagemakerPipeline(SagemakerPipeline):
    """Training sagemaker pipeline class"""

    def __init__(self) -> None:
        super().__init__()
        self.type = "train"
        self.pipeline_name = f"{self.project_name}-{self.type}-{self.env}"
        self.pipeline_description = (
            f"Training pipeline for training and tuning a model for {self.project_name}"
        )
        self.training_params = self.get_training_parameters()
        self.output_path = self.s3_paths["inference_test_path"]

        keys, _, _, _ = self.parse_modeling_schema(
            file_path=os.path.join(self.project_name, "data", "modeling_schema.json")
        )
        self.keys_length = len(keys)
        self.steps = TrainingSteps(
            session=self.pipeline_session, output_path=self.output_path
        )

    def get_pipeline_parameters(self):
        return {
            "S3TrainInput": self.training_config.get("train_input_data_csv"),
            "MaeThreshold": self.env_config.get(
                "sagemaker.training.accuracy_mae_threshold"
            ),
            "TrainInstanceCount": self.env_config.get(
                "sagemaker.training.train_instance_count"
            ),
            "MaxTuningJob": self.env_config.get("sagemaker.training.max_tuning_jobs"),
            "MaxParallelJobs": self.env_config.get(
                "sagemaker.training.max_parallel_jobs"
            ),
        }

    def get_training_parameters(self) -> dict:
        """Define the input parameters to be used in the sagemaker pipeline"""
        s3_train_input = ParameterString(name="S3TrainInput")
        accuracy_mae_threshold = ParameterFloat(
            name="MaeThreshold",
            default_value=self.env_config.get(
                "sagemaker.training.accuracy_mae_threshold"
            ),
        )

        # Tuning
        train_instance_count = ParameterInteger(
            name="TrainInstanceCount",
            default_value=self.env_config.get(
                "sagemaker.training.train_instance_count"
            ),
        )
        max_tuning_jobs = ParameterInteger(
            name="MaxTuningJob",
            default_value=self.env_config.get("sagemaker.training.max_tuning_jobs"),
        )
        max_parallel_jobs = ParameterInteger(
            name="MaxParallelJobs",
            default_value=self.env_config.get("sagemaker.training.max_parallel_jobs"),
        )

        register_new_data_quality_baseline = ParameterBoolean(
            name="RegisterNewDataQualityBaseline", default_value=True
        )

        return {
            "s3_train_input": s3_train_input,
            "accuracy_mae_threshold": accuracy_mae_threshold,
            "train_instance_count": train_instance_count,
            "max_tuning_jobs": max_tuning_jobs,
            "max_parallel_jobs": max_parallel_jobs,
            "register_new_data_quality_baseline": register_new_data_quality_baseline,
        }

    def get_sagemaker_pipeline(self) -> Pipeline:
        """Combine all steps into a pipeline

        Returns:
            Pipeline: The training sagemaker pipeline
        """
        self.logger.info(f"{self.env.upper()} Constructing training pipeline")

        # Input processor
        step_process = self.steps.get_input_processor_step(
            input_data=self.training_params.get("s3_train_input"),
            output_path=self.s3_paths["data_processed_path"],
        )

        step_data_baseline_registry = self.steps.get_baseline_data_drift_step(
            baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs[
                "train_no_target_with_header"
            ].S3Output.S3Uri,
            output_s3_uri=Join(
                values=[
                    self.s3_bucket,
                    self.s3_data_drift_monitoring,
                    "baseline",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                ],
                on="/",
            ),
        )

        ## Tuning
        step_lightgbm_tuning = self.steps.get_tune_model_step(
            train_data_s3_path=step_process.properties.ProcessingOutputConfig.Outputs[
                "lightgbm"
            ].S3Output.S3Uri,
            max_jobs=self.training_params["max_tuning_jobs"],
            max_parallel_jobs=self.training_params["max_parallel_jobs"],
        )
        step_lightgbm_tuning.add_depends_on([step_process])

        step_create_best_lightgbm, best_model = self.steps.get_create_model_step(
            model=step_lightgbm_tuning.get_top_model_s3_uri(
                top_k=0,
                s3_bucket=self.env_config.get("s3_bucket"),
                prefix=f"{self.env_config.get('s3_keys.main')}/models/{self.model_type}",
            )
        )

        ## Test predictions
        step_lightgbm_transform = self.steps.get_batch_transform_step(
            model_name=step_create_best_lightgbm.properties.ModelName,
            data_uri=Join(
                values=[
                    step_process.properties.ProcessingOutputConfig.Outputs[
                        "lightgbm"
                    ].S3Output.S3Uri,
                    "test",
                ],
                on="/",
            ),
            input_filter="$[1:]",  # Take out target column
        )

        step_lightgbm_transform.add_depends_on([step_create_best_lightgbm])

        ## Evaluate
        step_evaluate_lightgbm, evaluation_report = self.steps.get_evaluate_model_step(
            actuals_data_s3_uri=Join(
                values=[
                    step_process.properties.ProcessingOutputConfig.Outputs[
                        "lightgbm"
                    ].S3Output.S3Uri,
                    "test",
                ],
                on="/",
            ),
            inference_data_s3_uri=self.s3_paths["inference_test_path"],
            evaluation_script_path=f"{self.scripts_root_path}/evaluate.py",
        )

        step_evaluate_lightgbm.add_depends_on([step_lightgbm_transform])

        ## Register model
        # Create ModelMetrics object using the evaluation report from the evaluation step
        # A ModelMetrics object contains metrics captured from a model.
        model_metrics_lightgbm = self.get_model_metrics(
            evaluation_s3_path=Join(
                values=[
                    step_evaluate_lightgbm.arguments["ProcessingOutputConfig"][
                        "Outputs"
                    ][0]["S3Output"]["S3Uri"],
                    "evaluation.json",
                ],
                on="/",
            ),
        )

        drift_check_baselines = self.get_driftcheck_baselines(
            step_data_baseline_registry=step_data_baseline_registry
        )

        (
            step_register_lightgbm_model,
            step_register_lightgbm_model_approval,
        ) = self.steps.get_register_model_step(
            model=best_model,
            model_metrics=model_metrics_lightgbm,
            drift_check_baselines=drift_check_baselines,
            dependencies=[step_evaluate_lightgbm],
        )

        ## Config output
        step_config = self.steps.get_config_step(
            job_arguments=[
                "--config-file-s3-path",
                self.env_config.get("s3_keys.train_config"),
                "--training-date",
                ExecutionVariables.START_DATETIME,
                "--encoders-s3-path",
                step_process.properties.ProcessingOutputConfig.Outputs[
                    "encoders"
                ].S3Output.S3Uri,
                "--model-location",
                step_create_best_lightgbm.properties.PrimaryContainer.ModelDataUrl,
                "--baseline-data-statistics",
                step_data_baseline_registry.properties.CalculatedBaselineStatistics,
                "--baseline-data-constraints",
                step_data_baseline_registry.properties.CalculatedBaselineConstraints,
            ]
        )

        ## Accuracy condition
        step_cond = self.steps.get_condition_step(
            evaluation_step=step_evaluate_lightgbm,
            evaluation_report=evaluation_report,
            if_steps=[
                step_register_lightgbm_model_approval,
                step_config,
                step_data_baseline_registry,
            ],
            else_steps=[step_register_lightgbm_model],
            mae_threshold=self.training_params["accuracy_mae_threshold"],
        )

        ## Construct pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=list(self.training_params.values()),
            steps=[
                step_process,
                step_lightgbm_tuning,
                step_create_best_lightgbm,
                step_lightgbm_transform,
                step_evaluate_lightgbm,
                step_cond,
            ],
        )

        return pipeline

    def get_model_metrics(self, evaluation_s3_path) -> ModelMetrics:
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=evaluation_s3_path,
                content_type="application/json",
            ),
        )
        return model_metrics

    def get_driftcheck_baselines(
        self, step_data_baseline_registry: QualityCheckStep
    ) -> DriftCheckBaselines:
        drift_check_baselines = DriftCheckBaselines(
            model_data_statistics=MetricsSource(
                s3_uri=step_data_baseline_registry.properties.CalculatedBaselineStatistics,
                content_type="application/json",
            ),
            model_data_constraints=MetricsSource(
                s3_uri=step_data_baseline_registry.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
        )
        return drift_check_baselines


if __name__ == "__main__":
    print(TrainSagemakerPipeline().publish_sagemaker_pipeline())
    # print(TrainSagemakerPipeline().generate_pipeline_definition())
    # print(TrainSagemakerPipeline().trigger_sagemaker_pipeline())
