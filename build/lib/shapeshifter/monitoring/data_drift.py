"""
Code for data drift monitoring
"""
from sagemaker.model_monitor import CronExpressionGenerator, BatchTransformInput
from sagemaker.model_monitor.model_monitoring import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import MonitoringDatasetFormat

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline
from shapeshifter.utils.generic_functions import read_json_from_s3


class DataDriftMonitor(SagemakerPipeline):
    """
    Class for data drift monitoring pipeline creation
    """

    def __init__(self):
        super().__init__()
        self.training_config = read_json_from_s3(
            s3_proxy=self.s3_lib,
            s3_bucket=self.env_config.get("s3_bucket"),
            s3_path_key=self.env_config.get("s3_keys.train_config"),
        )

    def get_sagemaker_pipeline(self):
        pass

    def get_quality_check_step(
        self,
        check_job_config,
        quality_check_config,
    ):
        pass

    def create_data_drift_monitoring_schedule(self):
        """
        Function to create monitoring schedule for data drift detection
        """
        monitoring_schedule_name = f"data-drift-monitor-{self.env}"
        monitoring_schedule_summaries = self.sm_client.list_monitoring_schedules()[
            "MonitoringScheduleSummaries"
        ]

        if not any(
            monitoring_schedule.get("MonitoringScheduleName")
            == monitoring_schedule_name
            for monitoring_schedule in monitoring_schedule_summaries
        ):
            data_quality_model_monitor = DefaultModelMonitor(
                role=self.role,
                instance_count=1,
                instance_type="ml.c5.xlarge",
                volume_size_in_gb=120,
                base_job_name=f"{self.env}-data-drift-job",
                sagemaker_session=self.pipeline_session,
            )

            data_quality_model_monitor.create_monitoring_schedule(
                monitor_schedule_name=monitoring_schedule_name,
                batch_transform_input=BatchTransformInput(
                    data_captured_destination_s3_uri=f"{self.s3_bucket}/{self.s3_project_path}/data_capture",
                    destination=f"{self.root}input",
                    dataset_format=MonitoringDatasetFormat.csv(header=True),
                ),
                output_s3_uri=f"{self.s3_bucket}/{self.s3_project_path}/monitoring/data_drift_checks",
                statistics=self.training_config.get("baseline_data_statistics"),
                constraints=self.training_config.get("baseline_data_constraints"),
                schedule_cron_expression=CronExpressionGenerator.daily(hour=6),
                enable_cloudwatch_metrics=True,
            )


if __name__ == "__main__":
    pass
    # DataDriftMonitor().create_data_drift_monitoring_schedule()
