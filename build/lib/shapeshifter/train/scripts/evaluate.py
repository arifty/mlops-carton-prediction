"""Evaluation script for measuring mean squared error."""
import argparse
import json
import logging
import os
import pathlib
from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def generate_report(
    mean_squared_error: float,
    r2_score: float,
    mean_absolute_error: float,
    mean_absolute_percentage_error: float,
    file_name: str = "evaluation.json",
    output_dir: str = "/opt/ml/processing/evaluation",
):
    """Generate the report dictionary used to log the metrics in the registry

    Args:
        mean_squared_error (float): calculated MSE value
        r2_score (float): calculated R2 value
        mean_absolute_error (float): calculated MAE value
        mean_absolute_percentage_error (float): calculated MAPE value
        file_name (str, optional): name of the output file. Defaults to "evaluation.json".
        output_dir (str, optional): output directory to store the report in. Defaults to "/opt/ml/processing/evaluation".
    """
    report_dict = {
        "regression_metrics": {
            "mean_squared_error": {
                "value": mean_squared_error,
                "standard_deviation": "NaN",
            },
            "r2_score": {"value": r2_score, "standard_deviation": "NaN"},
            "mean_absolute_error": {
                "value": mean_absolute_error,
                "standard_deviation": "NaN",
            },
            "mean_absolute_percentage_error": {
                "value": mean_absolute_percentage_error,
                "standard_deviation": "NaN",
            },
        },
    }

    print("Save metrics")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    temp = pd.DataFrame.from_dict(report_dict)
    temp.to_json(os.path.join(output_dir, file_name), orient="columns")


def parse_modeling_schema(file_path: str) -> Tuple[list, list, str, dict]:
    """From the schema json, get all the different components

    Args:
        file_path (str): path to the modeling schema file

    Returns:
        Tuple[list, list, str, dict]: returns a list of keys, features and the target variable.
                                      Next to this also a dictionary with the dtypes for those elements.
    """
    with open(file_path) as f:
        modeling_schema = json.load(f)

    keys = list(modeling_schema.get("keys").keys())
    features = list(modeling_schema.get("features").keys())
    target = list(modeling_schema.get("target").keys())[0]

    dtypes = {}
    for element in ["keys", "features", "target"]:
        dtypes.update(modeling_schema.get(element).items())

    return keys, features, target, dtypes


if __name__ == "__main__":
    print("Starting evaluation")
    logger.debug("Starting evaluation.")
    base_dir = "/opt/ml/processing/input"
    prediction_folder = os.path.join(base_dir, "lightgbm")

    # Get actuals
    print("Read test data")
    keys, features, target, dtypes = parse_modeling_schema(
        file_path=os.path.join(base_dir, "schema", "modeling_schema.json")
    )

    df_actuals = pd.read_csv(
        os.path.join(base_dir, "data.csv"),
        names=[target] + features + keys,
    )

    # Read data from model XGBoost
    print("Read predictions")
    df_predictions = pd.read_csv(
        os.path.join(prediction_folder, "data.csv.out"),
        names=features + keys + ["PREDICTION"],
    )

    print("Merging actuals and predictions")
    df = df_actuals[keys + [target]].merge(
        df_predictions[keys + ["PREDICTION"]], on=keys
    )

    # Calculate metrics
    print("Calculate metrics")
    y_true = df[target].to_numpy()
    y_pred = df["PREDICTION"].to_numpy()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("Mean Squared Error: {}".format(mse))
    print("R2 Score: {}".format(r2))
    print("Mean Absolute Error: {}".format(mae))

    generate_report(
        mean_squared_error=mse,
        r2_score=r2,
        mean_absolute_error=mae,
        mean_absolute_percentage_error=mape,
    )
