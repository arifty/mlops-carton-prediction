"""
Snowflake query to delete old data of Inference data
"""
from shapeshifter.config.env_config import EnvConfig


def sf_inference_archive_data(env_config: EnvConfig, archival_date: str) -> str:
    db = env_config.get("snowflake.write_db")
    schema = env_config.get("snowflake.schema")
    inference_timestamp = env_config.get("snowflake.inference_timestamp")
    query = f"""
        DELETE FROM {db}.{schema}.SALES_ORDER_UNIT_CONVERSIONS
        WHERE {inference_timestamp} <= '{archival_date}'
    """
    return query
