"""Evaluation script for measuring mean squared error."""
import logging
import os
import pathlib

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

# Data
keys = [
    "SALES_ORDER_HEADER_NUMBER",
    "SALES_ORDER_ITEM_NUMBER",
    "SALES_ORDER_SCHEDULE_LINE_NUMBER",
]

features = [
    "CHANNEL_CLASS",
    "DISTRIBUTION_CHANNEL",
    "DIVISION_CODE",
    "SIZE_CODE",
    "SILHOUETTE_SHORT",
    "SALES_ORDER_ITEM_VAS_INDICATOR",
    "VAS_CODE_ZP1",
    "VAS_CODE_SK",
    "VAS_CODE_C20",
    "VAS_CODE_C4X",
    "VAS_CODE_PR",
    "VAS_CODE_C90",
    "VAS_CODE_STD",
    "VAS_CODE_CL1",
    "VAS_CODE_LBC",
    "VAS_CODE_SM",
    "VAS_CODE_CU",
    "VAS_CODE_ES",
    "VAS_CODE_C40",
    "VAS_CODE_CTU",
    "VAS_CODE_CLX",
    "VAS_CODE_SZU",
    "VAS_CODE_REST",
    "VAS_CODE_NONE",
    #     "SHIPPING_LOCATION_CODE",
    "COUNTRY_CODE",
    "CUSTOMER_ACCOUNT_GROUP_CODE",
    "SALES_ORDER_TYPE",
    "FULL_CASE_QUANTITY",
    "SOSL_TOTAL_QTY",
    "SOI_TOTAL_QUANTITY",
    "SOH_TOTAL_QUANTITY",
]

target = "NBR_CARTONS_RATIO"


if __name__ == "__main__":
    print("Starting evaluation")
    logger.debug("Starting evaluation.")

    base_dir = "/opt/ml/processing/input"
    test_folder = base_dir
    prediction_folder = os.path.join(base_dir, "lightgbm")

    # Get actuals
    print("Read test data")
    df_actuals = pd.read_csv(
        os.path.join(test_folder, "data.csv"), names=keys + [target] + features
    )

    # Read data from model XGBoost
    print("Read predictions")
    df_predictions = pd.read_csv(
        os.path.join(prediction_folder, "data.csv.out"),
        names=keys + features + ["PREDICTION"],
    )

    print(f"Length predictions: {df_predictions.shape[0]}")

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

    report_dict = {
        "regression_metrics": {
            "mean_squared_error": {"value": mse, "standard_deviation": "NaN"},
            "r2_score": {"value": r2, "standard_deviation": "NaN"},
            "mean_absolute_error": {"value": mae, "standard_deviation": "NaN"},
            "mean_absolute_percentage_error": {
                "value": mape,
                "standard_deviation": "NaN",
            },
        },
    }

    print("Save metrics")
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    temp = pd.DataFrame.from_dict(report_dict)
    temp.to_json(os.path.join(output_dir, "evaluation.json"), orient="columns")
