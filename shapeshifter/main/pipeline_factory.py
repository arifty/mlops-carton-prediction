"""
Factory for pipeline object creation
"""
import importlib

from shapeshifter.main.model import FlowType, RunType
from shapeshifter.utils.generic_functions import snake_case_to_camel


def pipeline_factory(
    run_type: RunType,
    flow_type: FlowType,
):
    """
    Function to generate data pipeline or core pipeline class from
    train or inference flows

    Args:
        run_type (RunType): either  train or inference runs
        flow_type (FlowType): either data or ml pipeline runs

    Returns:
        Pipeline class corresponding to the given flows
    """
    domain_camel = (
        f"{snake_case_to_camel(run_type.value)}{snake_case_to_camel(flow_type.value)}"
    )
    module = importlib.import_module(
        f"shapeshifter.pipelines.{run_type.value}.{flow_type.value}"
    )
    class_ = getattr(module, domain_camel)
    return class_
