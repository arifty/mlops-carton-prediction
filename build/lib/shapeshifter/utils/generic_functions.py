"""
Generic Helper functions
"""
import json
import logging.config
import re
from logging import Logger
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import boto3
from aws_utils.s3_lib import S3Proxy
import pandas as pd
import pandera as pa
from cerberus_utils.cerberus_reader import CerberusConfig
from snowflake_utils.snowflake_proxy import SnowflakeConfig, SnowflakeProxy
from tabulate import tabulate

from ..config.logging_config import LOGGING_CONFIG


def get_assume_role_credentials(role_arn: str) -> Dict:
    """fetch the crentials of the assumed role in aws

    Args:
        role_arn (str): roleArn of the running environment

    Returns:
        Dict: dictioanly with AccessKey, SecretAccess & Token
    """
    try:
        sts_client = boto3.client("sts")
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName="AssumeRoleSession",
        )
    except Exception as err:
        raise ConnectionError(f"Could not assume AWS role: {role_arn}") from err
    return assumed_role_object["Credentials"]


def parse_arg_dates(
    start_date: Optional[str], end_date: Optional[str]
) -> Tuple[str, str]:
    """
    Manages the dates received as arguments in the main function.

    Args:
        start_date (Optional[str]): Start Date
        end_date (Optional[str]): End Date

    Returns:
        (date, date): tuple containing the start date and the end date to use in the proper format
    """
    start_date = None if start_date == "" else start_date
    end_date = None if end_date == "" else end_date
    if end_date is None and start_date is None:
        end_date_: str = "9999-12-31"
        start_date_: str = (datetime.now() - timedelta(days=365 * 2)).strftime(
            "%Y-%m-%d"
        )

    elif end_date is None or start_date is None or end_date < start_date:
        raise ValueError(
            "You need to specify a specific range of dates with start date earlier than end date. On the contrary "
            "last month will be calculated. "
        )
    else:
        start_date_ = start_date
        end_date_ = end_date

    return start_date_, end_date_


def get_logger(
    logging_dict: Dict = LOGGING_CONFIG, logger_name: str = __name__
) -> Logger:
    """
    Get logger with predefined configuration

    Args:
        logging_dict (Dict, optional): Dictionary with logging configuration
        logger_name (str, optional): Defaults to "shapeshifter".

    Returns:
        logging: instantiated logger object
    """
    logging.config.dictConfig(logging_dict)
    return logging.getLogger(logger_name)


def camelcase_to_snake(word: str):
    """
    Convert a string from camel case to snake case

    Args:
        word (str): string to convert

    Returns:
        converted string
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", word).lower()


def snake_case_to_camel(word: str):
    """
    Convert a string from snake case to camel case

    Args:
        word (str): string to convert

    Returns:
        converted string
    """
    return "".join(w.title() for w in word.split("_"))


def update_json_in_s3(
    s3_proxy: S3Proxy,
    s3_bucket: str,
    s3_path_key: str,
    json_key: str,
    value_to_update: object,
) -> dict:
    """Updates a json file in the S3 location for the given key and value in it.
       If json doesn't exist, a new json will be created with the key and value.

    Args:
        s3_proxy (S3Proxy): instantiated object of the S3Proxy class
        s3_bucket (str): bucket name where the json file exists. Only the bucket name, not with complete key
        s3_path_key (str): S3 file path key (including file name), without the bucket name.
        json_key (str): json key for which value needs to be updated
        value_to_update (object): value to be updated.

    Returns:
        dict: updated json dictionary
    """
    json_content = read_json_from_s3(
        s3_proxy=s3_proxy,
        s3_bucket=s3_bucket,
        s3_path_key=s3_path_key,
        return_empty=True,
    )

    json_content[json_key] = value_to_update

    s3_proxy.put_object(
        obj=json.dumps(json_content),
        key=s3_path_key,
    )

    return json_content


def read_json_from_s3(
    s3_proxy: S3Proxy,
    s3_bucket: str,
    s3_path_key: str,
    return_empty: bool = False,
):
    """Read a json file from the S3 location provided.

    Args:
        s3_proxy (S3Proxy): instantiated object of the S3Proxy class
        s3_bucket (str): bucket name where the json file exists. Only the bucket name, not with complete key
        s3_path_key (str): S3 file path key (including file name), without the bucket name.
        return_empty (bool): True if you want to return an emtpy dictionary if the file does not exist, False if you want to raise an error.

    Returns:
        dict: dictionary with key value pairs from the json file
    """
    if s3_proxy.check_key_exists(key=s3_path_key):
        json_content = json.loads(
            s3_proxy.client.get_object(Bucket=s3_bucket, Key=s3_path_key)["Body"]
            .read()
            .decode("utf-8")
        )
    elif return_empty:
        json_content = {}
    else:
        raise FileNotFoundError(
            f"The requested file {s3_bucket}/{s3_path_key} was not found. Please provide an existing json file."
        )
    return json_content


def get_snowflake_writer(env_config) -> SnowflakeProxy:
    """
    Function to get snowflake connection with write permissions

    Args:
        env_config (Config): Environment based configuration

    Returns:
        SnowflakeProxy: snowflkae proxy connection
    """
    sf_config = SnowflakeConfig(
        role=env_config.get("snowflake.write_role"),
        warehouse=env_config.get("snowflake.write_warehouse"),
        schema=env_config.get("snowflake.schema"),
        database=env_config.get("snowflake.write_db"),
    )

    cerberus_config = CerberusConfig(user_key="user", password_key="password")
    sf_writer = SnowflakeProxy.from_cerberus(
        cerberus_config=cerberus_config,
        cerberus_sdb=env_config.get("cerberus.sdb_snowflake_write"),
        snowflake_config=sf_config,
    )
    return sf_writer


def validate_df_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    logger: Optional[Logger] = None,
) -> pd.DataFrame:

    """validate dataframe for pandera schema and raise

    Args:
        df (pd.DataFrame): input dataframe
        schema (pa.DataFrameSchema): schema to be validated with
        logger (Logger, optional): instantiated Logger if exists. Defaults to None.

    Returns:
        pd.DataFrame: updated dataframe
    """
    logger = get_logger(logging_dict=LOGGING_CONFIG) if logger is None else logger
    try:
        df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as pandera_error:
        logger.error("Data error: Schema errors found")

        pandera_failures = pandera_error.failure_cases[
            ["column", "check", "failure_case", "index"]
        ]
        pandera_error_tab = tabulate(
            pandera_failures,
            tablefmt="grid",
            showindex=False,
            headers=pandera_failures.columns,
        )

        logger.error(pandera_error_tab)
        raise
    except Exception as err:
        logger.error("Other validation errors found!!")
        raise err
    return df
