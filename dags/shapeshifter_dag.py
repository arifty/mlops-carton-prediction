import os
from datetime import datetime, timedelta
from typing import Optional

from airflow import DAG
from airflow.models import Variable
from generic.custom_operator import CustomKubernetesPodOperator
from kubernetes.client import models as k8s

DATA_FLOW_TYPE = "input_data_pipeline"
ML_FLOW_TYPE = "sagemaker_pipeline"
OUTPUT_FLOW_TYPE = "output_data_pipeline"
airflow_namespace = os.getenv("AIRFLOW__KUBERNETES__NAMESPACE")
env_airflow = Variable.get("env")
cluster_id = Variable.get("cluster_id")
shapeshifter_core_image = Variable.get("shapeshifter_image")
env_var_1 = k8s.V1EnvVar(name="ENV", value=env_airflow)
sf_warehouse_size_original = Variable.get(f"knnights_snowflake_original_warehouse_size")
sf_warehouse_size_scaled = Variable.get(f"shapeshifter_snowflake_scaled_warehouse_size")
shapeshifter_start_date = Variable.get("shapeshifter_start_date", default_var=None)
shapeshifter_end_date = Variable.get("shapeshifter_end_date", default_var=None)
shapeshifter_schedule = (
    None
    if Variable.get("shapeshifter_schedule", default_var=None) == ""
    else Variable.get("shapeshifter_schedule", default_var=None)
)


def create_dag(run_type: str, dag_id: str, schedule: Optional[str], default_args: dict):
    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        schedule_interval=schedule,
        catchup=False,
    )

    with dag:
        sf_wh_scaling_up_task = CustomKubernetesPodOperator(
            task_id="sf_wh_scaling_up_task",
            name="sf-pod",
            image=shapeshifter_core_image,
            cmds=[
                "python",
                "-m",
                "shapeshifter.utils.snowflake_warehouse_scaling",
                "--size",
                sf_warehouse_size_scaled,
            ],
        )

        data_pipeline_task = CustomKubernetesPodOperator(
            task_id="data_pipeline_task",
            name="shapeshifter-pod",
            image=shapeshifter_core_image,
            cmds=[
                "python",
                "-m",
                "shapeshifter.main.run_pipeline",
                "--run_type",
                run_type,
                "--flow_type",
                DATA_FLOW_TYPE,
                "--start_date",
                shapeshifter_start_date,
                "--end_date",
                shapeshifter_end_date,
            ],
            env_vars=[env_var_1],
            resources={"request_memory": "12300M"},
            log_events_on_failure=False,
        )

        sf_wh_scaling_down_task = CustomKubernetesPodOperator(
            task_id="sf_wh_scaling_down_task",
            name="sf-pod",
            image=shapeshifter_core_image,
            cmds=[
                "python",
                "-m",
                "shapeshifter.utils.snowflake_warehouse_scaling",
                "--size",
                sf_warehouse_size_original,
            ],
        )

        ml_pipeline_task = CustomKubernetesPodOperator(
            task_id="ml_pipeline_task",
            name="shapeshifter-pod",
            image=shapeshifter_core_image,
            cmds=[
                "python",
                "-m",
                "shapeshifter.main.run_pipeline",
                "--run_type",
                run_type,
                "--flow_type",
                ML_FLOW_TYPE,
            ],
            env_vars=[env_var_1],
        )

        if run_type == "inference":
            output_pipeline_task = CustomKubernetesPodOperator(
                task_id="output_pipeline_task",
                name="shapeshifter-pod",
                image=shapeshifter_core_image,
                cmds=[
                    "python",
                    "-m",
                    "shapeshifter.main.run_pipeline",
                    "--run_type",
                    run_type,
                    "--flow_type",
                    OUTPUT_FLOW_TYPE,
                ],
                env_vars=[env_var_1],
                resources={"request_memory": "12300M"},
            )

            (
                sf_wh_scaling_up_task
                >> data_pipeline_task
                >> sf_wh_scaling_down_task
                >> ml_pipeline_task
                >> output_pipeline_task
            )
        else:
            (
                sf_wh_scaling_up_task
                >> data_pipeline_task
                >> sf_wh_scaling_down_task
                >> ml_pipeline_task
            )

    return dag


for RUN_TYPE in ["train", "inference"]:
    default_args = {
        "description": f"Shapeshifter {RUN_TYPE} pipeline dag",
        "owner": "Airflow",
        "depends_on_past": False,
        "start_date": datetime(2022, 6, 24),
        "email": ["ELC.AdvancedAnalytics@Nike.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    }
    dag_id = f"shapeshifter-{RUN_TYPE}-pipeline"
    schedule = shapeshifter_schedule if RUN_TYPE == "inference" else None
    globals()[dag_id] = create_dag(
        run_type=RUN_TYPE, dag_id=dag_id, schedule=schedule, default_args=default_args
    )
