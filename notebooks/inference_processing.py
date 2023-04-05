import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

dtypes = {
    "SALES_ORDER_HEADER_NUMBER": str,
    "SALES_ORDER_ITEM_NUMBER": str,
    "SALES_ORDER_SCHEDULE_LINE_NUMBER": str,
    "CHANNEL_CLASS": str,
    "DISTRIBUTION_CHANNEL": str,
    "DIVISION_CODE": str,
    "PRODUCT_CODE": str,
    "SIZE_CODE": str,
    "SILHOUETTE_SHORT": str,
    "SALES_ORDER_ITEM_VAS_INDICATOR": str,
    "VAS_CODE_ZP1": np.int32,
    "VAS_CODE_SK": np.int32,
    "VAS_CODE_C20": np.int32,
    "VAS_CODE_C4X": np.int32,
    "VAS_CODE_PR": np.int32,
    "VAS_CODE_C90": np.int32,
    "VAS_CODE_STD": np.int32,
    "VAS_CODE_CL1": np.int32,
    "VAS_CODE_LBC": np.int32,
    "VAS_CODE_SM": np.int32,
    "VAS_CODE_CU": np.int32,
    "VAS_CODE_ES": np.int32,
    "VAS_CODE_C40": np.int32,
    "VAS_CODE_CTU": np.int32,
    "VAS_CODE_CLX": np.int32,
    "VAS_CODE_SZU": np.int32,
    "VAS_CODE_REST": np.int32,
    "VAS_CODE_NONE": np.int32,
    "COUNTRY_CODE": str,
    "CUSTOMER_ACCOUNT_GROUP_CODE": str,
    "SALES_ORDER_TYPE": str,
    "FULL_CASE_QUANTITY": np.float64,
    "SOSL_TOTAL_QTY": np.float64,
    "SOI_TOTAL_QUANTITY": np.float64,
    "SOH_TOTAL_QUANTITY": np.float64,
    "NBR_CARTONS": np.int64,
    "NBR_CARTONS_RATIO": np.float64,
}

# Data
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
    "COUNTRY_CODE",
    "CUSTOMER_ACCOUNT_GROUP_CODE",
    "SALES_ORDER_TYPE",
    "FULL_CASE_QUANTITY",
    "SOSL_TOTAL_QTY",
    "SOI_TOTAL_QUANTITY",
    "SOH_TOTAL_QUANTITY",
]

keys = [
    "SALES_ORDER_HEADER_NUMBER",
    "SALES_ORDER_ITEM_NUMBER",
    "SALES_ORDER_SCHEDULE_LINE_NUMBER",
]

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    input_folder = os.path.join(base_dir, "input")
    encoders_folder = os.path.join(input_folder, "encoders")
    input_data_folder = os.path.join(input_folder, "data")

    output_folder = os.path.join(base_dir, "output")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("Reading input data")
    folders = glob.glob(f"{input_data_folder}/[!config]*/", recursive=True)
    df = pd.read_csv(f"{max(folders)}csv/inference_data.csv", dtype=dtypes)

    categorical_cols = df[features].select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].astype(str)

    print("Start encoding")
    for cat in categorical_cols:
        print(f"Encoding {cat}")
        le = joblib.load(f"{encoders_folder}/{cat}_encoder.joblib")

        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        df[cat] = df[cat].apply(lambda x: le_dict.get(x, None))

    print("Saving data")
    df[keys + features].to_csv(f"{output_folder}/data.csv", index=False, header=False)
    print(df[keys + features].dtypes)
