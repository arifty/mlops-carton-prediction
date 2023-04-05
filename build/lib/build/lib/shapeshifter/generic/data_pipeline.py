"""
Generic shapeshifter data pipeline
"""
from abc import ABC, abstractmethod
from datetime import datetime
import boto3

from aws_utils.s3_lib import S3Proxy
import pyspark.sql.functions as f
from cerberus_utils.cerberus_reader import CerberusConfig
from pyspark.sql import DataFrame

from ..config.env_config import EnvConfig
from ..config.logging_config import LOGGING_CONFIG
from ..utils import generic_functions, spark_helper


class DataPipeline(ABC):
    """
    Class containing data processing steps and functions that are common per use case for data pipelines.
    """

    def __init__(
        self,
        s3_lib: S3Proxy = None,
    ):
        """
        Init class
        """
        self.logger = generic_functions.get_logger(
            logging_dict=LOGGING_CONFIG, logger_name=__name__
        )
        self.env_config = EnvConfig()
        self.token = self.env_config.get("token")
        self.env = self.env_config.get("env")
        class_name = self.__class__.__name__
        self.run_type = generic_functions.camelcase_to_snake(class_name).replace(
            "_data_pipeline", ""
        )
        self.cerberus_config = CerberusConfig(
            token=self.env_config.get("token"),
            user_key=self.env_config.get("cerberus.okta_user_key"),
            password_key=self.env_config.get("cerberus.okta_pwd_key"),
        )
        self.curr_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self.s3_output_config_key = self.env_config.get(
            f"s3_keys.{self.run_type.replace('_input', '').replace('_output', '')}_config"
        )
        self.data_config_key = self.run_type + "_data"

        self.date_partition_column = "SALES_ORDER_HEADER_DOCUMENT_DATE"
        self.col_prefix = "header_document_"
        session = boto3.Session()
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn="arn:aws:iam::188961369755:role/cet-commercial-emea0-euwest1-knnightscicdrole",
            RoleSessionName="run_pipeline",
        )
        print(response)
        print("successfully assumed role.")
        self.spark = spark_helper.get_or_create_spark_session(
            self.env_config.get("sagemaker_role")
        )
        self.s3_lib = (
            S3Proxy(
                bucket=self.env_config.get("s3_bucket"),
                endpoint_url=self.env_config.get("aws.s3_endpoint"),
            )
            if s3_lib is None
            else s3_lib
        )

    def add_date_partition(self, df: DataFrame) -> DataFrame:
        """
        Function to add year and month columns to the given data frame.

        Args:
            df (DataFrame): data frame to add the date columns

        Returns (DataFrame):
            data frame with the columns year & month added

        """
        df = df.withColumn(
            f"{self.col_prefix}year",
            f.lit(f.year(f.to_date(f.col(self.date_partition_column), "yyyy-MM-dd"))),
        ).withColumn(
            f"{self.col_prefix}month",
            f.lit(f.month(f.to_date(f.col(self.date_partition_column), "yyyy-MM-dd"))),
        )
        self.logger.info("Adding date columns step finished")
        return df

    @abstractmethod
    def run(
        self,
        start_date: str,
        end_date: str,
    ):
        """
        Abstract run to update the data in snowflake.

        Args:
        """
        raise NotImplementedError
