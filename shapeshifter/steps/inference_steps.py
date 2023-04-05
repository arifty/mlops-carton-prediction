from typing import Optional, Union

from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import ProcessingStep

from shapeshifter.steps.generic_steps import SagemakerSteps


class InferenceSteps(SagemakerSteps):
    """Steps used for the inference pipeline"""

    def __init__(self, session, output_path: str) -> None:
        super().__init__(session, output_path)

    def get_input_processor_step(
        self,
        data_path: Union[str, PipelineVariable],
        encoders_path: Union[str, PipelineVariable],
    ) -> ProcessingStep:
        ## Get data
        step_process = ProcessingStep(
            name="PrepInferenceData",
            processor=self.get_sklearn_processor(),
            code=f"{self.scripts_root_path}/inference_input_processing.py",
            inputs=[
                ProcessingInput(
                    source=data_path,
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
                data_path,
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
            cache_config=self.cache_config,
        )

        return step_process

    def get_quality_check_step(
        self,
        baseline_data_statitics: Optional[Union[str, PipelineVariable]],
        baseline_data_constraints: Optional[Union[str, PipelineVariable]],
        check_job_config: CheckJobConfig,
        quality_check_config: DataQualityCheckConfig,
        register_new_baseline: Union[bool, PipelineVariable] = False,
    ) -> QualityCheckStep:
        step_data_drift_monitor = QualityCheckStep(
            name="DataDriftMonitoring",
            display_name="data-drift-monitoring",
            description="Monitoring step for data drift detection",
            check_job_config=check_job_config,
            quality_check_config=quality_check_config,
            skip_check=register_new_baseline,
            supplied_baseline_statistics=baseline_data_statitics,
            supplied_baseline_constraints=baseline_data_constraints,
            fail_on_violation=False,
        )

        return step_data_drift_monitor

    def get_model_step(self, model_location: Union[str, PipelineVariable]):
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
            sagemaker_session=self.session,
            role=self.role,
        )

        step_create_best_lightgbm = ModelStep(
            name="GetBestLightGBM",
            step_args=lgbm_model.create(
                instance_type=self.env_config.get("sagemaker.inference_instance_type"),
                tags=self.tags,
            ),
        )

        return step_create_best_lightgbm
