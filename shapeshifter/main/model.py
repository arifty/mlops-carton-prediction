"""
Pydantic models for shapeshifter
"""
from enum import Enum


class RunType(str, Enum):
    """
    Allowed run types
    """

    train = "train"
    inference = "inference"


class FlowType(str, Enum):
    """
    Allowed flow types
    """

    input_data_pipeline = "input_data_pipeline"
    sagemaker_pipeline = "sagemaker_pipeline"
    output_pipeline = "output_data_pipeline"
