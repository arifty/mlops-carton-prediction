from datetime import datetime, timedelta

import pandas as pd
import pandera as pa
from aws_utils.s3_lib import S3Proxy

from shapeshifter.data.queries.sf_inference_archive_data import (
    sf_inference_archive_data,
)
from shapeshifter.data.queries.sf_inference_latest_date import sf_inference_latest_date
from shapeshifter.data.schemas import (
    inference_s3_output_columns,
    inference_sf_output_columns,
)
from shapeshifter.generic.output_data_pipeline import OutputDataPipeline


class InferenceOutputDataPipeline(OutputDataPipeline):
    """
    Class containing data pipeline steps for inferencing run
    """

    def __init__(self, s3_lib: S3Proxy = None):
        super().__init__(s3_lib)
        self.inference_timestamp = self.env_config.get("snowflake.inference_timestamp")
        self.creation_timestamp = self.env_config.get("snowflake.creation_timestamp")
        self.ts_format = "%Y-%m-%d %H:%M:%S"
        self.__s3_output_key = f"{self.env_config.get('s3_keys.inference')}/live"

    def get_latest_date(self) -> str:
        """
        Getter for the specific query for this process.

        Args:

        Returns:
            latest date (str): snowflake table data for max prediction timestamp
        """

        sf_latest_date = self.snowflake_writer.fetch_df(
            query=sf_inference_latest_date(self.env_config)
        )[self.inference_timestamp][0]
        return sf_latest_date.strftime(self.ts_format)

    def execute_archive_data(self, latest_date: str) -> pd.DataFrame:
        """
        Executor for the specific query for archival of old data.

        Args:
            latest_date (str): to find latest_date - archival days

        Returns:
            pd.DataFrame : snowflake response after query execution
        """
        sf_archive_date = (
            datetime.strptime(latest_date, self.ts_format)
            - timedelta(days=self.env_config.get("snowflake.archive_lookback"))
        ).strftime(self.ts_format)

        return self.snowflake_writer.execute_queries(
            sf_inference_archive_data(self.env_config, sf_archive_date)
        )

    def read_latest_files_s3(self, latest_date: str) -> list:
        """Get a list of keys to be read with a modified date later than the latest processing date"""
        s3_files = []
        try:
            contents = self.s3_lib.list_objects_with_timestamp(
                prefix=self.__s3_output_key,
            )
            for obj in contents:
                last_modified = obj["Timestamp"].strftime(self.ts_format)
                if datetime.strptime(last_modified, self.ts_format) > datetime.strptime(
                    latest_date, self.ts_format
                ):
                    s3_files.append([obj["Key"], last_modified])
        except KeyError as err:
            raise KeyError(
                f"Error reading files in this S3 path: {self.__s3_output_key},"
            ) from err
        return s3_files

    def read_inference_output_data(
        self, s3_files: list, s3_output_columns_schema: pa.DataFrameSchema
    ) -> pd.DataFrame:
        """read and prepare the csv data from inference output

        Args:
            s3_files (list[str, str]): List of [s3_key, modified_timestamp]
            s3_output_columns_schema (pa.DataFrameSchema): s3 columns schema

        Returns:
            pd.DataFrame: prepared inference data in dataframe
        """
        csv_columns = [x.upper() for x in list(s3_output_columns_schema.columns.keys())]
        df_inference = pd.DataFrame(columns=csv_columns + [self.creation_timestamp])

        for s3_key, modified_dt in s3_files:
            check_csv = ".".join(s3_key.split("/")[-1].split(".")[-2:])
            if check_csv == "csv.out":
                s3_path = (
                    f"{self.env_config.get('aws.s3_endpoint')}/"
                    + f"{self.env_config.get('wi_s3_bucket')}/"
                    + f"{s3_key}"
                )
                self.logger.info(f"Reading csv from S3: {s3_path}")
                df_csv = pd.read_csv(s3_path, names=csv_columns, header=0)
                df_csv[self.inference_timestamp] = (
                    pd.to_datetime(modified_dt, utc=True)
                    .tz_convert(None)
                    .strftime(self.ts_format)
                )

                # COMMENTING OUT VALIDATION, DUE TO POD MEMORY ISSUES
                # df_csv = validate_df_schema(
                #     df_csv, s3_output_columns_schema, self.logger
                # )
                # adding the creation timestamp column to final df
                df_csv[self.creation_timestamp] = datetime.now().strftime(
                    self.ts_format
                )
                df_inference = pd.concat([df_inference, df_csv], ignore_index=True)
        return df_inference

    def run(
        self,
        start_date: str,
        end_date: str,
    ):
        """
        Run different steps to retrieve output data and update the data in snowflake.

        Args:
        """
        self.logger.info("Shapeshifter inference output data pipeline run started.")
        sf_latest_date = self.get_latest_date()
        self.logger.info(f"Snowflake table latest inference date: {sf_latest_date}")

        # Retrieve inference data from S3 (if lastmodified is greater than last predictions)
        all_files = self.read_latest_files_s3(latest_date=sf_latest_date)
        df_inference = self.read_inference_output_data(
            s3_files=all_files, s3_output_columns_schema=inference_s3_output_columns
        )

        self.logger.info("Start archiving old data from snowflake")
        self.execute_archive_data(sf_latest_date)

        df_inference_write = df_inference[
            [*[x.upper() for x in inference_sf_output_columns]]
        ]
        self.logger.info(
            f"Nr. of rows to be updated in snowflake: {len(df_inference_write.index)}"
        )

        self.snowflake_writer.write_df(
            df=df_inference_write,
            table=self.env_config.get("snowflake.inference_table").upper(),
            database=self.env_config.get("snowflake.write_db").upper(),
            schema=self.env_config.get("snowflake.schema").upper(),
        )
        self.logger.info("Completed uploading to snowflake")
