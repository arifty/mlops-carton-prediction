import re
import subprocess
import sys
from typing import Tuple

# Install additional package
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-U", "pandas", "--quiet"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-U", "numpy", "--quiet"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "feature-engine", "--quiet"]
)


import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from feature_engine.encoding import RareLabelEncoder

TRAIN_PERC = 0.6
TEST_PERC = 0.2


def apply_rare_encoding(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Apply rare encoding to wanted features and stores the encoder on the file system.

    Args:
        df (pd.DataFrame): Dataframe with the features that need to be rare encoded
        features (list): List of features to rare encode

    Returns:
        pd.DataFrame: Original dataframe with the required features rare encoded
    """
    for feature in features:
        df[feature] = df[feature].cat.add_categories("Rare").fillna("Rare")

    rare_encoder = RareLabelEncoder(tol=0.001, replace_with="Rare")

    rare_encoder.fit(df[features])
    df[features] = rare_encoder.transform(df[features])

    joblib.dump(rare_encoder, f"{encoders_folder}/rare_encoder.joblib", compress=9)

    return df


def apply_label_encoding(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Apply label encoding and store the encoders on the file system.

    Args:
        df (pd.DataFrame): Dataframe with the features that need to be label encoded
        features (list): List of features that have to be label encoded

    Returns:
        pd.DataFrame: Original dataframe with the required features label encoded
    """
    le = LabelEncoder()
    for cat in features:
        df[cat] = le.fit_transform(df[cat])
        joblib.dump(le, f"{encoders_folder}/{cat}_encoder.joblib", compress=9)

    return df


def split_data(
    df: pd.DataFrame, split_key: str, train_perc: float, test_perc: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe in a train, validation and test dataset based on a splitting key.
    All records with the same key will be kept together in one of the three resulting datasets.

    Args:
        df (pd.DataFrame): Input dataset that has to be split
        split_key (str): The key that defines which records need to be kept together
        train_perc (float): the percentage of trianing records
        test_perc (float): the percentage of test records. The leftover part will be used for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
    """
    splitter = GroupShuffleSplit(test_size=(1 - train_perc), n_splits=2, random_state=1)
    split = splitter.split(df, groups=df[split_key])
    train_inds, rest_inds = next(split)

    train_df = df.iloc[train_inds]
    rest_df = df.iloc[rest_inds]

    splitter = GroupShuffleSplit(
        test_size=round(test_perc / (1 - train_perc), 2), n_splits=2, random_state=1
    )
    split = splitter.split(rest_df, groups=rest_df[split_key])
    validation_inds, test_inds = next(split)

    validation_df = rest_df.iloc[validation_inds]
    test_df = rest_df.iloc[test_inds]

    return train_df, validation_df, test_df


def create_lightgbm_folder(root_folder: str) -> str:
    """Create the folder for lightgbm input files

    Args:
        root_folder (str): parent folder

    Returns:
        str: the lightgbm folder path
    """
    lightgbm_folder = os.path.join(root_folder, "lightgbm")
    Path(os.path.join(lightgbm_folder, "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(lightgbm_folder, "validation")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(lightgbm_folder, "test")).mkdir(parents=True, exist_ok=True)
    return lightgbm_folder


def create_encoders_folder(root_folder: str) -> str:
    """Create the folder for the encoder files

    Args:
        root_folder (str): parent folder

    Returns:
        str: the encoders folder path
    """
    encoders_folder = os.path.join(root_folder, "encoders")
    Path(encoders_folder).mkdir(parents=True, exist_ok=True)
    return encoders_folder


def create_train_without_header_folder(root_folder: str) -> str:
    """Create the folder for the training data without headers

    Args:
        root_folder (str): parent folder

    Returns:
        str: the folder path for the training data
    """
    train_no_target_with_header_folder = os.path.join(
        root_folder, "train_no_target_with_header"
    )
    Path(train_no_target_with_header_folder).mkdir(parents=True, exist_ok=True)
    return train_no_target_with_header_folder


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
    base_dir = "/opt/ml/processing"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-s3-filename", type=str, dest="input_s3_file_name")
    parser.add_argument(
        "--rare-features",
        type=lambda s: re.split(" |, ", s),
        dest="rare_features",
    )
    args = parser.parse_args()

    input_folder = os.path.join(base_dir, "input")
    processed_folder = os.path.join(base_dir, "processed")
    lightgbm_folder = create_lightgbm_folder(root_folder=processed_folder)
    encoders_folder = create_encoders_folder(root_folder=processed_folder)
    train_no_target_with_header_folder = create_train_without_header_folder(
        root_folder=processed_folder
    )

    print("Reading dataset")
    keys, features, target, dtypes = parse_modeling_schema(
        file_path=os.path.join(input_folder, "schema", "modeling_schema.json")
    )

    df = pd.read_csv(
        os.path.join(input_folder, args.input_s3_file_name.split("/")[-1]),
        dtype=dtypes,
    )

    print("Encoding categorical variables")
    df = apply_rare_encoding(df=df, features=args.rare_features)

    # Label encode all categorical features
    categorical_cols = (
        df[features].select_dtypes(include=["object", "category"]).columns
    )
    df = apply_label_encoding(df=df, features=categorical_cols)

    print("Splitting data")
    # Split the dataset into train, test and validation
    train_df, validation_df, test_df = split_data(
        df=df,
        split_key="SALES_ORDER_HEADER_NUMBER",
        train_perc=TRAIN_PERC,
        test_perc=TEST_PERC,
    )

    print(
        f"Train record count: {train_df.shape[0]}\nValidation record count: {validation_df.shape[0]}\nTest record count: {test_df.shape[0]}"
    )

    print("Apply log transformation to target on training and validation data")
    # We don't apply this on the test data as we want to have the original actuals there.
    train_df[target] = np.log(train_df[target])
    validation_df[target] = np.log(validation_df[target])

    print("Storing data on s3")
    # Upload train and test set to S3
    train_df[features].to_csv(
        f"{train_no_target_with_header_folder}/data.csv", index=False, header=True
    )

    train_df[[target] + features].to_csv(
        f"{lightgbm_folder}/train/data.csv", index=False, header=False
    )
    validation_df[[target] + features].to_csv(
        f"{lightgbm_folder}/validation/data.csv", index=False, header=False
    )
    test_df[[target] + features + keys].to_csv(
        f"{lightgbm_folder}/test/data.csv", index=False, header=False
    )

    print("Create categorical index json file")
    with open(f"{lightgbm_folder}/categorical_index.json", "w") as outfile:
        json.dump(
            {
                "cat_index_list": [
                    train_df[[target] + features].columns.get_loc(c)
                    for c in categorical_cols
                    if c in train_df[[target] + features]
                ]
            },
            outfile,
        )
