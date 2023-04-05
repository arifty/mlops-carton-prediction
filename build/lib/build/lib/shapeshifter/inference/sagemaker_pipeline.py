"""
Inference Sagemaker pipeline.
"""

from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import ProcessingStep

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline
from shapeshifter.utils.generic_functions import read_json_from_s3


class InferenceSagemakerPipeline(SagemakerPipeline):
    """Inference sagemaker pipeline class"""

    def __init__(self) -> None:
        super().__init__()
        self.type = "inference"
        self.pipeline_name = f"{self.project_name}-{self.type}-{self.env}"
        self.pipeline_description = f"{self.env.upper()} {self.type} pipeline for batch predictions for {self.project_name}"
        self.output_path = self.s3_paths["inference_live_path"]

    @property
    def inference_config(self):
        return read_json_from_s3(
            s3_proxy=self.s3_lib,
            s3_bucket=self.env_config.get("s3_bucket"),
            s3_path_key=self.env_config.get("s3_keys.inference_config"),
        )

    def get_pipeline_parameters(self):
        return {
            "s3_input_data": self.inference_config.get("inference_input_data_csv"),
            "encoders_path": self.training_config.get("encoders_s3_path"),
            "model_location": self.training_config.get("model_location"),
        }

    def get_quality_check_step(
        self,
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
    ) -> QualityCheckStep:
        step_data_drift_monitor = QualityCheckStep(
            name="DataDriftMonitoring",
            display_name="data-drift-monitoring",
            description="Monitoring step for data drift detection",
            check_job_config=check_job_config,
            quality_check_config=quality_check_config,
            skip_check=False,
            supplied_baseline_statistics=self.baseline_data_statitics,
            supplied_baseline_constraints=self.baseline_data_constraints,
            fail_on_violation=False,
        )

        return step_data_drift_monitor

    def get_sagemaker_pipeline(self) -> Pipeline:
        """Generate sagemaker pipeline for inferencing.

        Returns:
            Pipeline: inference pipeline object
        """
        self.logger.info(f"{self.env.upper()} Constructing pipeline")
        model_location = ParameterString("model_location")
        s3_input_data = ParameterString("s3_input_data")
        encoders_path = ParameterString("encoders_path")
        self.baseline_data_statitics = ParameterString("baseline_data_statitics")
        self.baseline_data_constraints = ParameterString("baseline_data_constraints")

        ## Get data
        sklearn_processor = self.get_sklearn_processor()

        step_process = ProcessingStep(
            name="PrepInferenceData",
            processor=sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=s3_input_data,
                    destination=Join(values=[self.root, "input/data"]),
                ),
                ProcessingInput(
                    source=encoders_path,
                    destination=Join(values=[self.root, "input/encoders"]),
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
                s3_input_data,
                "--rare-features",
                f"{self.rare_features_string}",
            ],
            outputs=[
                ProcessingOutput(
                    output_name="inference_data",
                    source=f"{self.root}output",
                    destination=Join(
                        values=[
                            self.s3_bucket,
                            self.env_config.get("s3_keys.inference_input"),
                        ],
                        on="/",
                    ),
                ),
                ProcessingOutput(
                    output_name="inference_with_header",
                    source=f"{self.root}output_with_header",
                    destination=Join(
                        values=[
                            self.s3_bucket,
                            self.env_config.get("s3_keys.inference_input_with_header"),
                        ],
                        on="/",
                    ),
                ),
            ],
            code="shapeshifter/inference/scripts/inference_processing.py",
            cache_config=self.cache_config,
        )

        # Step 3: Create a model package

        ### DO NOT DELETE
        ## Commented code is preferred way of working, though currently not possible.
        ## AWS team is working on a resolution.
        # lgbm_model = ModelPackage(
        #     role=role,
        #     model_package_arn=model_arn,
        #     source_dir="model_scripts",
        #     entry_point="inference.py",
        #     sagemaker_session=pipeline_session,
        # )

        lgbm_model = Model(
            model_data=model_location,
            image_uri=self.env_config.get("sagemaker.docker_image_inference"),
            sagemaker_session=self.pipeline_session,
            role=self.role,
        )

        step_create_best_lightgbm = ModelStep(
            name="GetBestLightGBM",
            step_args=lgbm_model.create(
                instance_type=self.env_config.get("sagemaker.inference_instance_type"),
                tags=self.tags,
            ),
        )

        step_lightgbm_batch_transform = self.get_batch_transform_step(
            model_name=step_create_best_lightgbm.properties.ModelName,
            data_uri=Join(
                values=[
                    step_process.properties.ProcessingOutputConfig.Outputs[
                        "inference_data"
                    ].S3Output.S3Uri,
                    "/",
                ],
                on="",
            ),
            generate_inference_id=True,
        )

        step_data_drift_monitoring = self.get_baseline_data_drift_step(
            baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs[
                "inference_with_header"
            ].S3Output.S3Uri,
            output_s3_uri=Join(
                on="/",
                values=[
                    self.s3_bucket,
                    self.s3_data_drift_monitoring,
                    "violations",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                ],
            ),
        )

        ## Construct pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[
                step_process,
                step_create_best_lightgbm,
                step_data_drift_monitoring,
                step_lightgbm_batch_transform,
            ],
            parameters=[
                s3_input_data,
                encoders_path,
                model_location,
                self.baseline_data_statitics,
                self.baseline_data_constraints,
            ],
        )

        return pipeline


if __name__ == "__main__":
    InferenceSagemakerPipeline().publish_sagemaker_pipeline()
    # InferenceSagemakerPipeline().trigger_sagemaker_pipeline()
