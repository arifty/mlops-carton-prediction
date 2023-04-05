"""
Code to run the pipeline for a given type; inference, train, data_pipeline, ml_pipeline
Example:  python -m shapeshifter.main.run_pipeline --run_type inference
          --flow_type data_pipeline --start_date 2022-02-23 --end_date 2022-02-24
"""
import argparse

from shapeshifter.main.model import FlowType, RunType
from shapeshifter.main.pipeline_factory import pipeline_factory
from shapeshifter.utils.generic_functions import parse_arg_dates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", help="Run types, e.g; train, inference")
    parser.add_argument(
        "--flow_type",
        help="Flow types, e.g; data_pipeline, ml_pipeline, output_pipeline",
    )
    parser.add_argument("--start_date", help="Start Date")
    parser.add_argument("--end_date", help="End Date")
    args = parser.parse_args()

    start_date, end_date = parse_arg_dates(args.start_date, args.end_date)
    class_ = pipeline_factory(RunType[args.run_type], FlowType[args.flow_type])
    pipeline_class = class_()
    pipeline_class.run(start_date, end_date)
