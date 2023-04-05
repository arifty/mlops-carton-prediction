import io
import logging
import os
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> lightgbm.Booster:
    """Read model saved in model_dir and return a object of lightgbm model.

    Args:
        model_dir (str): directory that saves the model artifact.

    Returns:
        obj: lightgbm model.
    """
    try:
        return joblib.load(os.path.join(model_dir, "model.pkl"))
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(
    task: lightgbm.Booster, input_data: Any, content_type: str, accept: str
) -> np.array:  # type: ignore
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        task (lightgbm.Booster): model loaded by model_fn.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form
            (response_data, content_type)
    """
    if content_type == "text/csv":
        data = pd.read_csv(io.StringIO(input_data), sep=",")
        data.columns = [f"feature_{x}" for x in range(data.shape[1])]
        try:
            model_output = task.predict(
                data.iloc[:, :-3], num_iteration=task.best_iteration
            )
            # Reverse log transformation
            data["PREDICTION"] = np.exp(model_output)
            print("Prediction added to dataframe")
            if accept.endswith(";verbose"):
                accept = accept.rstrip(";verbose")
            return encoder.encode(data.to_numpy(), accept)
        except Exception:
            logging.exception("Failed to do transform")
            raise
    if content_type == "application/x-parquet":
        data = pd.read_parquet(input_data)
        data.columns = [f"feature_{x}" for x in range(data.shape[1])]
        try:
            model_output = task.predict(
                data.iloc[:, 3:], num_iteration=task.best_iteration
            )
            # Reverse log transformation
            data["PREDICTION"] = np.exp(model_output)
            print("Prediction added to dataframe")
            if accept.endswith(";verbose"):
                accept = accept.rstrip(";verbose")
            return encoder.encode(data.to_numpy(), accept)
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError(
        '{{"error": "unsupported content type {}"}}'.format(content_type or "unknown")
    )
