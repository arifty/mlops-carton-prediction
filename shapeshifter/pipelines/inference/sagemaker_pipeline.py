"""
Inference Sagemaker pipeline.
"""

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline
from shapeshifter.steps.inference_steps import InferenceSteps
from shapeshifter.utils.generic_functions import read_json_from_s3


class InferenceSagemakerPipeline(SagemakerPipeline):
    """Inference sagemaker pipeline class"""

    def __init__(self) -> None:
        super().__init__()
        self.type = "inference"
        self.pipeline_name = f"{self.project_name}-{self.type}-{self.env}"
        self.pipeline_description = f"{self.env.upper()} {self.type} pipeline for batch predictions for {self.project_name}"
        self.output_path = self.s3_paths["inference_live_path"]
        self.steps = InferenceSteps(
            session=self.pipeline_session, output_path=self.output_path
        )

    @property
    def inference_config(self):
        return read_json_from_s3(
            s3_proxy=self.s3_lib,
            s3_bucket=self.env_config.get("wi_s3_bucket"),
            s3_path_key=self.env_config.get("s3_keys.inference_config"),
        )

    def get_pipeline_parameters(self):
        return {
            "s3_input_data": self.inference_config.get("inference_input_data_csv"),
            "encoders_path": self.training_config.get("encoders_s3_path"),
            "model_location": self.training_config.get("model_location"),
        }

    def get_sagemaker_pipeline(self) -> Pipeline:
        """Generate sagemaker pipeline for inferencing.

        Returns:
            Pipeline: inference pipeline object
        """
        self.logger.info(f"{self.env.upper()} Constructing pipeline")
        model_location = ParameterString("model_location")
        s3_input_data = ParameterString("s3_input_data")
        encoders_path = ParameterString("encoders_path")
        baseline_data_statitics = ParameterString("baseline_data_statitics")
        baseline_data_constraints = ParameterString("baseline_data_constraints")

        step_process = self.steps.get_input_processor_step(
            data_path=s3_input_data, encoders_path=encoders_path
        )

        step_create_best_lightgbm = self.steps.get_model_step(
            model_location=model_location
        )

        step_lightgbm_batch_transform = self.steps.get_batch_transform_step(
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

        step_data_drift_monitoring = self.steps.get_baseline_data_drift_step(
            baseline_data_statitics=baseline_data_statitics,
            baseline_data_constraints=baseline_data_constraints,
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
                baseline_data_statitics,
                baseline_data_constraints,
            ],
        )

        return pipeline


if __name__ == "__main__":
    InferenceSagemakerPipeline().publish_sagemaker_pipeline()
    # InferenceSagemakerPipeline().trigger_sagemaker_pipeline()
