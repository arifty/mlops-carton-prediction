"""
Generic shapeshifter export pipeline
"""
from abc import abstractmethod

from aws_utils.s3_lib import S3Proxy

from shapeshifter.generic.data_pipeline import DataPipeline
from ..utils import generic_functions


class OutputDataPipeline(DataPipeline):
    """
    Class containing export data steps and functions that are common per use case for output pipelines.
    """

    def __init__(
        self,
        s3_lib: S3Proxy = None,
    ):
        """
        Init class
        """
        super().__init__()
        self.snowflake_writer = generic_functions.get_snowflake_writer(self.env_config)

    @abstractmethod
    def get_latest_date(
        self,
    ) -> str:
        """
        Abstract getter for the specific query.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_archive_data(self, latest_date: str) -> str:
        """
        Abstract executor for the specific query for archival.
        """
        raise NotImplementedError

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
