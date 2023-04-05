"""
Training Sagemaker pipeline
"""
import os
from typing import List, Tuple
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import ProcessingStep, TuningStep, Step

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline


class TrainSagemakerPipeline(SagemakerPipeline):
    """Training sagemaker pipeline class"""

    def __init__(self) -> None:
        super().__init__()
        self.type = "train"
        self.pipeline_name = f"{self.project_name}-{self.type}-{self.env}"
        self.pipeline_description = (
            f"Training pipeline for training and tuning a model for {self.project_name}"
        )
        self.tuning_instance_type = self.env_config.get(
            "sagemaker.tuning_instance_type"
        )
        self.inference_instance_type = self.env_config.get(
            "sagemaker.inference_instance_type"
        )
        self.training_params = self.get_training_parameters()
        self.output_path = self.s3_paths["inference_test_path"]

        keys, _, _, _ = self.parse_modeling_schema(
            file_path=os.path.join(self.project_name, "data", "modeling_schema.json")
        )
        self.keys_length = len(keys)

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

        # data ingestion processor
        step_data_ingestion = self.get_data_processor_step(load_type="Input")
        # Input processor
        step_process = self.get_input_processor_step()

        step_data_baseline_registry = self.get_baseline_data_drift_step(
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
        step_lightgbm_tuning = self.get_tune_model_step(step_process)

        step_create_best_lightgbm, best_model = self.get_create_model_step(
            model=step_lightgbm_tuning.get_top_model_s3_uri(
                top_k=0,
                s3_bucket=self.env_config.get("s3_bucket"),
                prefix=f"{self.env_config.get('s3_keys.main')}/models/{self.model_type}",
            )
        )

        ## Test predictions
        step_lightgbm_transform = self.get_batch_transform_step(
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
            generate_inference_id=False,
            input_filter="$[1:]",  # Take out target column
        )

        step_lightgbm_transform.add_depends_on([step_create_best_lightgbm])

        ## Evaluate
        step_evaluate_lightgbm, evaluation_report = self.get_evaluate_model_step(
            data_s3_uri=Join(
                values=[
                    step_process.properties.ProcessingOutputConfig.Outputs[
                        "lightgbm"
                    ].S3Output.S3Uri,
                    "test",
                ],
                on="/",
            ),
            evaluation_script_path="shapeshifter/train/scripts/evaluate.py",
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
        ) = self.get_register_model_step(
            model=best_model,
            model_metrics=model_metrics_lightgbm,
            drift_check_baselines=drift_check_baselines,
            dependencies=[step_evaluate_lightgbm],
        )

        ## Config output
        step_config = self.get_config_step(
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
        step_cond = self.get_condition_step(
            evaluation_step=step_evaluate_lightgbm,
            evaluation_report=evaluation_report,
            if_steps=[
                step_register_lightgbm_model_approval,
                step_config,
                step_data_baseline_registry,
            ],
            else_steps=[step_register_lightgbm_model],
        )

        ## Construct pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=list(self.training_params.values()),
            steps=[
                step_data_ingestion,
                step_process,
                step_lightgbm_tuning,
                step_create_best_lightgbm,
                step_lightgbm_transform,
                step_evaluate_lightgbm,
                step_cond,
            ],
        )

        return pipeline

    def get_data_processor_step(self, load_type: str = "Input") -> ProcessingStep:
        """Define the data ingestion processor"""
        step_processor = self.get_pyspark_processor()
        step_args = step_processor.run(
            submit_app=f"{self.project_name}/train/scripts/data_ingestion.py",
            submit_py_files=[
                f"{self.project_name}/code/hello_py_spark_udfs.py",
                f"{self.project_name}/config/__init__.py",
                f"{self.project_name}/config/env_config.py",
                f"{self.project_name}/config/env_config.conf",
                f"{self.project_name}/config/logging_config.py",
                f"{self.project_name}/main/__init__.py",
                f"{self.project_name}/main/model.py",
                f"{self.project_name}/main/pipeline_factory.py",
                f"{self.project_name}/main/run_pipeline.py",
                f"{self.project_name}/generic/__init__.py",
                f"{self.project_name}/generic/data_pipeline.py",
                f"{self.project_name}/generic/input_data_pipeline.py",
                f"{self.project_name}/train/__init__.py",
                f"{self.project_name}/train/input_data_pipeline.py",
            ],
            logs=False,
        )

        step_process = ProcessingStep(name=f"{load_type}Data", step_args=step_args)
        return step_process

    def get_input_processor_step(self) -> ProcessingStep:
        """Define the input processor"""
        step_process = ProcessingStep(
            name="PrepData",
            processor=self.get_sklearn_processor(),
            inputs=[
                ProcessingInput(
                    source=self.training_params.get("s3_train_input"),
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
                self.training_params.get("s3_train_input"),
                "--rare-features",
                f"{self.rare_features_string}",
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train_no_target_with_header",
                    source=f"{self.root}processed/train_no_target_with_header",
                    destination=Join(
                        values=[
                            self.s3_paths["data_processed_path"],
                            "train_no_target_with_header",
                        ],
                        on="/",
                    ),
                ),
                ProcessingOutput(
                    output_name=self.model_type,
                    source=f"{self.root}processed/{self.model_type}",
                    destination=Join(
                        values=[self.s3_paths["data_processed_path"], "lightgbm"],
                        on="/",
                    ),
                ),
                ProcessingOutput(
                    output_name="encoders",
                    source=f"{self.root}processed/encoders",
                    destination=Join(
                        values=[self.s3_paths["data_processed_path"], "encoders"],
                        on="/",
                    ),
                ),
            ],
            code=f"{self.project_name}/train/scripts/processing.py",
            cache_config=self.cache_config,
        )
        return step_process

    def get_tune_model_step(
        self,
        dependant_step: ProcessingStep,
    ) -> TuningStep:
        """Define the tuning object for tuning the model

        Args:
            dependant_step (ProcessingStep): previous preprocessing step, from whoch output
                                              s3 of the training dataset is fetched

        Returns:
            TuningStep: Definition of the tuning step
        """
        train_data_s3_path = dependant_step.properties.ProcessingOutputConfig.Outputs[
            "lightgbm"
        ].S3Output.S3Uri

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
            sagemaker_session=self.pipeline_session,
        )

        metric_definitions = [
            {
                "Name": objective_metric_name,
                "Regex": f"{objective_metric_name}: ([0-9\.]+)",
            },
        ]

        tuner = HyperparameterTuner(
            estimator=lightgbm_estimator,
            objective_metric_name=objective_metric_name,
            objective_type="Minimize",
            metric_definitions=metric_definitions,
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=self.training_params["max_tuning_jobs"],
            max_parallel_jobs=self.training_params["max_parallel_jobs"],
            tags=self.tags,
        )

        step_lightgbm_tuning = TuningStep(
            name="HPTuning",
            step_args=tuner.fit(inputs=TrainingInput(s3_data=train_data_s3_path)),
        )
        step_lightgbm_tuning.add_depends_on([dependant_step])
        return step_lightgbm_tuning

    def get_create_model_step(self, model: Join) -> Tuple[ModelStep, PyTorchModel]:
        """Create the model step

        Args:
            model (Join): s3 uri to the model object
        """
        best_lightgbm_model = PyTorchModel(
            name="LightGBM",
            image_uri=self.env_config.get("sagemaker.docker_image_inference"),
            model_data=model,
            source_dir=f"{self.project_name}/train/scripts/inference_code",
            sagemaker_session=self.pipeline_session,
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

    def get_quality_check_step(
        self,
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
    ) -> QualityCheckStep:
        data_quality_check_step = QualityCheckStep(
            name="BaselineDataCreationStep",
            skip_check=True,
            register_new_baseline=self.training_params.get(
                "register_new_data_quality_baseline"
            ),
            quality_check_config=quality_check_config,
            check_job_config=check_job_config,
            model_package_group_name=f"{self.project_name}-{self.model_type}",
            cache_config=self.cache_config,
            display_name="data_baseline_creation",
            description="Baseline creation for data drift checks",
        )

        return data_quality_check_step

    def get_evaluate_model_step(
        self, data_s3_uri: str, evaluation_script_path: str
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
            inputs=[
                ProcessingInput(
                    source=data_s3_uri,
                    destination=f"{self.root}input",
                ),
                ProcessingInput(
                    source=self.s3_paths["inference_test_path"],
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
                        values=[self.s3_paths["inference_test_path"], "evaluation"],
                        on="/",
                    ),
                ),
            ],
            code=evaluation_script_path,
            property_files=[evaluation_report],
        )

        return step_evaluate_lightgbm, evaluation_report

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
            "model_package_group_name": f"{self.project_name}-{self.env}-{self.model_type}",
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
            outputs=[
                ProcessingOutput(
                    source=f"{self.root}config",
                    destination=Join(
                        values=[
                            self.s3_bucket,
                            self.env_config.get("s3_keys.config_root"),
                        ],
                        on="/",
                    ),
                ),
            ],
            job_arguments=job_arguments,
            code="shapeshifter/train/scripts/create_config.py",
            cache_config=self.cache_config,
        )
        return step_config

    def get_condition_step(
        self,
        evaluation_step: ProcessingStep,
        evaluation_report: PropertyFile,
        if_steps: List[Step],
        else_steps: List[Step],
    ) -> ConditionStep:
        # Create accuracy condition to ensure the model meets performance requirements.
        # Models with a test accuracy lower than the condition will not be registered with the model registry.
        cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="regression_metrics.mean_absolute_error.value",
            ),
            right=self.training_params["accuracy_mae_threshold"],
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


if __name__ == "__main__":
    print(TrainSagemakerPipeline().publish_sagemaker_pipeline())
    # print(TrainSagemakerPipeline().generate_pipeline_definition())
    # print(TrainSagemakerPipeline().trigger_sagemaker_pipeline())
