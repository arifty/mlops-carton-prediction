"""
PySpark Helper functions
"""
import os
from typing import Dict

import boto3
from cerberus_utils.cerberus_reader import Cerberus, CerberusConfig
from pyspark.sql import DataFrame, SparkSession

from shapeshifter.config.env_config import EnvConfig
from shapeshifter.utils import generic_functions


def get_or_create_spark_session(aws_role: str = "") -> SparkSession:
    """
    Function to get or create spark session from given configurations.

    Args:

    Returns:
        spark session with given configurations

    """
    logger = generic_functions.get_logger(logger_name=__name__)
    logger.info(
        f"assumed role for spark is: {boto3.client('sts').get_caller_identity()}"
    )
    if SparkSession.getActiveSession() is not None:
        return SparkSession.builder.getOrCreate()

    if not os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False) and aws_role == "":
        spark = (
            SparkSession.builder.master("local[*]")
            .config(
                "spark.driver.extraJavaOptions",
                "-Dhttp.proxyHost=127.0.0.1 \
                -Dhttp.proxyPort=9000 \
                -Dhttps.proxyHost=127.0.0.1 \
                -Dhttps.proxyPort=9000",
            )
            .config(
                "spark.jars.packages",
                "net.snowflake:spark-snowflake_2.12:2.10.1-spark_3.2,net.snowflake:snowflake-jdbc:3.13.14,com.amazonaws:aws-java-sdk-s3:1.11.901,org.apache.hadoop:hadoop-aws:3.2.3,com.google.guava:guava:27.0-jre",
            )
            .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:4566")
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config(
                "spark.executorEnv.AWS_PROFILE",
                "localstack",
            )
            .getOrCreate()
        )
    elif os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False) and aws_role == "":
        spark = (
            SparkSession.builder.master("local[*]")
            .config(
                "spark.jars.packages",
                "net.snowflake:spark-snowflake_2.12:2.10.1-spark_3.2,net.snowflake:snowflake-jdbc:3.13.14,com.amazonaws:aws-java-sdk-s3:1.11.901,org.apache.hadoop:hadoop-aws:3.2.3,com.google.guava:guava:27.0-jre",
            )
            .config(
                "spark.hadoop.fs.s3a.assumed.role.credentials.provider",
                "com.amazonaws.auth.AWSStaticCredentialsProvider",
            )
            .getOrCreate()
        )
    else:
        spark = (
            SparkSession.builder.master("local[4]")
            .config(
                "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
            )
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            .config(
                "spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored",
                "true",
            )
            .config("spark.hadoop.parquet.enable.summary-metadata", "false")
            .config("spark.sql.parquet.mergeSchema", "false")
            .config("spark.sql.parquet.filterPushdown", "true")
            .config("spark.sql.hive.metastorePartitionPruning", "true")
            .config(
                "spark.jars.ivy",
                "/tmp/.ivy",
            )
            .config(
                "spark.jars.packages",
                "net.snowflake:spark-snowflake_2.12:2.10.1-spark_3.2,net.snowflake:snowflake-jdbc:3.13.14,com.amazonaws:aws-java-sdk-s3:1.11.901,org.apache.hadoop:hadoop-aws:3.2.3,com.google.guava:guava:27.0-jre",
            )
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.auth.AssumedRoleCredentialProvider",
            )
            .config(
                "spark.hadoop.fs.s3a.assumed.role.credentials.provider",
                "com.amazonaws.auth.WebIdentityTokenCredentialsProvider",
            )
            .config(
                "spark.hadoop.fs.s3a.assumed.role.arn",
                aws_role,
            )
            .config(
                "spark.executorEnv.AWS_WEB_IDENTITY_TOKEN_FILE",
                "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
            )
            .config(
                "spark.executorEnv.AWS_ROLE_ARN",
                aws_role,
            )
            .getOrCreate()
        )

    logger.info("spark conf:")
    logger.info(f"{spark.sparkContext.getConf().getAll()}")
    spark.sparkContext.setLogLevel("WARN")

    return spark


def spark_sf_query(
    spark: SparkSession,
    env_config: EnvConfig,
    query: str,
    cerberus_user_key: str = "user",
    cerberus_pwd_key: str = "password",
) -> DataFrame:
    """
    Function to run snowflake query in spark and return data frame with preconfigured settings.

    Args:
        spark (SparkSession): spark session
        env_config (EnvConfig): environment configurations
        query (str): snowflake query to select data
        cerberus_user_key (str): key for user in cerberus
        cerberus_pwd_key (str): key for password in cerberus

    Returns (DataFrame):
        data frame returned by snowflake query

    """
    sf_config = env_config.get("snowflake")

    logger = generic_functions.get_logger(logger_name=__name__)
    logger.info("logging session details:")
    try:
        logger.info(boto3.client("sts").get_caller_identity())
    except Exception as err:
        logger.info(f"error: {err}")

    cerberus_config = CerberusConfig(token=env_config.get("token"), aws_session=None)
    cerberus_credentials = Cerberus(cerberus_config).get_credentials(
        sdb=env_config.get("cerberus.sdb_snowflake_read"),
    )

    sf_user = cerberus_credentials[cerberus_user_key]
    sf_password = cerberus_credentials[cerberus_pwd_key]

    sf_options = {
        "sfURL": "nike.snowflakecomputing.com",
        "sfAccount": "nike",
        "sfWarehouse": sf_config.get("read_warehouse"),
        "sfDatabase": sf_config.get("read_db"),
        "sfSchema": sf_config.get("schema"),
        "sfUser": sf_user,
        "sfPassword": sf_password,
        "sfAuthenticator": "https://nike.okta.com",
        "sfRole": sf_config.get("read_role"),
        "parallelism": "8",
        "purge": "on",
        "usestagingtable": "off",
    }
    # TODO: Fix and get it from env_config
    # os.environ["HTTPS_PROXY"] = env_config.get("https_proxy")
    os.environ["HTTPS_PROXY"] = ""
    # TODO: move config to dag !!!!!!

    logger.debug(query)
    df = (
        spark.read.format("net.snowflake.spark.snowflake")
        .options(**sf_options)
        .option("query", query)
        .load()
    )

    return df


def rename_df_columns(df: DataFrame, mapping_dict: Dict) -> DataFrame:
    """
    Rename columns of spark data frame

    Args:
        df (DataFrame): spark data frame to rename columns
        mapping_dict (Dict): dictionary with old column names as keys
                             and new column names as values

    Returns (DataFrame):
        spark data frame with columns renamed

    """
    diff_cols = list(set(mapping_dict.keys()) - set(df.columns))
    if diff_cols:
        raise NameError(f"{diff_cols} columns not in data frame")
    # else:
    new_column_names_list = list(
        map(lambda col: mapping_dict.get(col, col), df.columns)
    )
    df = df.toDF(*new_column_names_list)
    return df
