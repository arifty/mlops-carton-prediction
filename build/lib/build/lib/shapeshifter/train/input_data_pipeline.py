from typing import List

from ..data.queries.sf_train_data_query import sf_train_so_line_data_query
from ..generic.input_data_pipeline import InputDataPipeline


class TrainInputDataPipeline(InputDataPipeline):
    """
    Class containing data pipeline steps for training run
    """

    @staticmethod
    def get_query(start_date: str, end_date: str) -> str:
        """
        Getter for the specific query for this process.

        Args:
            start_date (str): start date to use for filtering data
            end_date (str): end date to use for filtering data

        Returns:
            query (str): snowflake query to execute for fetching data
        """
        return sf_train_so_line_data_query(start_date, end_date)
