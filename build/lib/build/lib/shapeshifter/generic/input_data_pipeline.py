from abc import abstractmethod
from typing import List
import boto3

from pyspark.sql import DataFrame

from shapeshifter.utils import generic_functions

from shapeshifter.generic.data_pipeline import DataPipeline
from shapeshifter.utils import spark_helper


class InputDataPipeline(DataPipeline):
    def __init__(
        self,
    ):
        """
        Init class
        """
        super().__init__()
        self.__s3_endpoint_bucket = f"{self.env_config.get('aws.s3_endpoint')}/{self.env_config.get('s3_bucket')}"
        self.__s3_path_key = (
            self.env_config.get(f"s3_keys.{self.run_type}_data")
            + f"{self.curr_timestamp}"
        )

    @staticmethod
    def get_partition_by_write_columns() -> List[str]:
        """
        Getter for the attribute partition_by_write_columns

        Returns:
            partition_by_write_columns (List[str]): column names to partition by when saving data frame
                (year, month, day will always be added to these by default)
        """
        return []

    @staticmethod
    @abstractmethod
    def get_query(start_date: str, end_date: str) -> str:
        """
        Abstract Getter for the specific query for this process.
        """
        raise NotImplementedError

    def run(
        self,
        start_date: str,
        end_date: str,
    ):
        """
        Run different steps to retrieve data, prepare and cleanse.
        Finally the data set will be saved.

        Args:
            start_date (str): start date to use when filtering data set
            end_date (str): end date to use when filtering data set
        """
        self.logger.info(f"Shapeshifter {self.run_type} data pipeline run started.")

        df = spark_helper.spark_sf_query(
            spark=self.spark,
            env_config=self.env_config,
            query=self.get_query(start_date, end_date),
        )
        df = self.add_date_partition(df)
        check_config_update = self.save_df_s3(df=df, file_type="csv")
        # COMMENTING-OUT writing parquet files as not needed for the moment
        # check_config_update = self.save_df_s3(df=df, file_type="parquet")
        self.logger.info(f"writing to s3 location: {self.s3_output_config_key}")

        if not check_config_update:
            raise Exception(f"Failed to run the data_pipeline!!!")
        return df

    def save_df_s3(self, df: DataFrame, file_type: str = "csv") -> dict:
        """Generic function to save df to s3 bucket, either csv or parquet files

        Args:
            df (DataFrame): pyspark dataframe to be saved
            file_type (str, optional): format (csv or parquet) to save. Defaults to "csv".

        Raises:
            err: Exception while saving to s3

        Returns:
            dict: saved json data, in case to do unit testing
        """
        partition_by_write_columns: List[str] = []
        partition_by_specific_cols = self.get_partition_by_write_columns()
        partition_by_date_cols: List[str] = [
            self.col_prefix + "year",
            self.col_prefix + "month",
        ]
        file_full_path = f"{self.__s3_endpoint_bucket}/{self.__s3_path_key}/{file_type}"

        self.logger.info("logging session details before writing to s3:")
        try:
            self.logger.info(boto3.client("sts").get_caller_identity())
        except Exception as err:
            self.logger.info(f"error: {err}")

        if file_type != "csv":
            partition_by_write_columns = (
                partition_by_specific_cols + partition_by_date_cols
            )

        try:
            df.coalesce(1).write.partitionBy(partition_by_write_columns).format(
                file_type
            ).mode("overwrite").option("header", "true").save(file_full_path)

            self.logger.info(f"Data saved successfully to S3: {file_full_path}")
            # get the filename (s3 key) of the randomly created csv file
            if file_type == "csv":
                csv_path_key = self.s3_lib.list_objects(
                    prefix=f"{self.__s3_path_key}/{file_type}", suffix=f".{file_type}"
                )[0]
                file_full_path = f"{self.__s3_endpoint_bucket}/{csv_path_key}"

            check_config_update = generic_functions.update_json_in_s3(
                s3_proxy=self.s3_lib,
                s3_bucket=self.env_config.get("s3_bucket"),
                s3_path_key=self.s3_output_config_key,
                json_key=f"{self.data_config_key}_{file_type}",
                value_to_update=file_full_path.replace(
                    self.env_config.get("aws.s3_endpoint"), "s3:/"
                ),
            )

        except Exception as err:
            self.logger.warn(f"Data could not be saved to S3: {file_full_path}!!")
            raise err

        return check_config_update
