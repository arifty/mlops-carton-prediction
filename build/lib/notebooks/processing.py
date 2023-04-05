import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

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
    "SHIPPING_LOCATION_CODE": str,
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

keys = [
    "SALES_ORDER_HEADER_NUMBER",
    "SALES_ORDER_ITEM_NUMBER",
    "SALES_ORDER_SCHEDULE_LINE_NUMBER",
]

train_perc = 0.6
test_perc = 0.2

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    input_folder = os.path.join(base_dir, "input")
    processed_folder = os.path.join(base_dir, "processed")
    train_folder = os.path.join(processed_folder, "train")
    validation_folder = os.path.join(processed_folder, "validation")
    test_folder = os.path.join(processed_folder, "test")
    test_no_target_folder = os.path.join(processed_folder, "test_no_target")
    test_with_header_folder = os.path.join(processed_folder, "test_with_header")
    lightgbm_folder = os.path.join(processed_folder, "lightgbm")
    encoders_folder = os.path.join(processed_folder, "encoders")

    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    Path(encoders_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(os.path.join(input_folder, "data.csv"), dtype=dtypes)

    print("Encoding categorical variables")
    categorical_cols = df[features].select_dtypes(include=["object"]).columns

    df[categorical_cols] = df[categorical_cols].astype(str)
    le = LabelEncoder()
    for cat in categorical_cols:
        df[cat] = le.fit_transform(df[cat])
        joblib.dump(le, f"{encoders_folder}/{cat}_encoder.joblib", compress=9)

    print("Splitting data")
    # Split the dataset into train, test and validation
    splitter = GroupShuffleSplit(test_size=(1 - train_perc), n_splits=2, random_state=1)
    split = splitter.split(df, groups=df["SALES_ORDER_HEADER_NUMBER"])
    train_inds, rest_inds = next(split)

    train_df = df.iloc[train_inds]
    rest_df = df.iloc[rest_inds]

    splitter = GroupShuffleSplit(
        test_size=round(test_perc / (1 - train_perc), 2), n_splits=2, random_state=1
    )
    split = splitter.split(rest_df, groups=rest_df["SALES_ORDER_HEADER_NUMBER"])
    validation_inds, test_inds = next(split)

    validation_df = rest_df.iloc[validation_inds]
    test_df = rest_df.iloc[test_inds]

    print(
        f"Train record count: {train_df.shape[0]}\nValidation record count: {validation_df.shape[0]}\nTest record count: {test_df.shape[0]}"
    )

    # creating a new directory called pythondirectory
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(validation_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)
    Path(test_no_target_folder).mkdir(parents=True, exist_ok=True)
    Path(test_with_header_folder).mkdir(parents=True, exist_ok=True)
    Path(lightgbm_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(lightgbm_folder, "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(lightgbm_folder, "validation")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(lightgbm_folder, "test")).mkdir(parents=True, exist_ok=True)

    print("Storing data on s3")
    # Upload train and test set to S3
    train_df[[target] + features].to_csv(
        f"{train_folder}/data.csv", index=False, header=False
    )
    validation_df[[target] + features].to_csv(
        f"{validation_folder}/data.csv", index=False, header=False
    )
    test_df[keys + [target] + features].to_csv(
        f"{test_folder}/data.csv", index=False, header=False
    )
    test_df[keys + features].to_csv(
        f"{test_no_target_folder}/data.csv", index=False, header=False
    )
    test_df.to_csv(f"{test_with_header_folder}/data.csv", index=False)

    train_df[[target] + features].to_csv(
        f"{lightgbm_folder}/train/data.csv", index=False, header=False
    )
    validation_df[[target] + features].to_csv(
        f"{lightgbm_folder}/validation/data.csv", index=False, header=False
    )
    test_df[[target] + features].to_csv(
        f"{lightgbm_folder}/test/data.csv", index=False, header=False
    )
