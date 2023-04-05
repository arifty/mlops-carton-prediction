import json
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
    [sys.executable, "-m", "pip", "install", "feature-engine==1.5", "--quiet"]
)

import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_modeling_schema(file_path: str) -> Tuple[list, list, dict]:
    """From the schema json, get all the different components

    Args:
        file_path (str): path to the modeling schema file

    Returns:
        Tuple[list, list, dict]: returns a list of keys, features and a dictionary
                                 with the dtypes for those two lists
    """
    with open(file_path) as f:
        modeling_schema = json.load(f)

    keys = list(modeling_schema.get("keys").keys())
    features = list(modeling_schema.get("features").keys())

    dtypes = {}
    for element in ["keys", "features"]:
        dtypes.update(modeling_schema.get(element).items())

    return keys, features, dtypes


def apply_rare_encoding(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Apply rare encoding to wanted features using the created encoders during training

    Args:
        df (pd.DataFrame): Dataframe with the features that need to be rare encoded
        features (list): List of features to rare encode

    Returns:
        pd.DataFrame: Original dataframe with the required features rare encoded
    """
    re = joblib.load(f"{encoders_folder}/rare_encoder.joblib")
    for feature in features:
        df[feature] = df[feature].cat.add_categories("Rare").fillna("Rare")

    df[features] = re.transform(df[features])

    return df


def apply_label_encoding(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Apply label encoding and store the encoders on the file system.

    Args:
        df (pd.DataFrame): Dataframe with the features that need to be label encoded
        features (list): List of features that have to be label encoded

    Returns:
        pd.DataFrame: Original dataframe with the required features label encoded
    """
    for cat in features:
        print(f"Encoding {cat}")
        le = joblib.load(f"{encoders_folder}/{cat}_encoder.joblib")

        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        df[cat] = df[cat].apply(lambda x: le_dict.get(x, None))

    return df


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
    encoders_folder = os.path.join(input_folder, "encoders")
    input_data_folder = os.path.join(input_folder, "data")

    output_folder = os.path.join(base_dir, "output")
    output_with_header_folder = os.path.join(base_dir, "output_with_header")

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(output_with_header_folder).mkdir(parents=True, exist_ok=True)

    print("Reading input data")
    keys, features, dtypes = parse_modeling_schema(
        file_path=os.path.join(input_folder, "schema", "modeling_schema.json")
    )

    df = pd.read_csv(
        os.path.join(input_data_folder, args.input_s3_file_name.split("/")[-1]),
        dtype=dtypes,
    )

    categorical_cols = (
        df[features].select_dtypes(include=["object", "category"]).columns
    )

    print("Start encoding")
    print("Apply rare encoding")
    df = apply_rare_encoding(df=df, features=args.rare_features)

    print("Apply label encoding")
    df = apply_label_encoding(df=df, features=categorical_cols)

    print("Saving data")
    df[features + keys].to_csv(f"{output_folder}/data.csv", index=False, header=False)
    df[features].to_csv(
        f"{output_with_header_folder}/data.csv", index=False, header=True
    )

    print(df[features + keys].dtypes)
