from typing import List

from shapeshifter.data.queries.sf_inference_data_query import sf_so_line_data_query
from shapeshifter.generic.input_data_pipeline import InputDataPipeline


class InferenceInputDataPipeline(InputDataPipeline):
    """
    Class containing data pipeline steps for inferencing run
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
        return sf_so_line_data_query(start_date, end_date)
