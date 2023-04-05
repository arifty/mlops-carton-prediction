"""
Snowflake query to fetch data for Inference data pipeline
"""
from shapeshifter.config.env_config import EnvConfig


def sf_inference_latest_date(env_config: EnvConfig) -> str:
    db = env_config.get("snowflake.write_db")
    schema = env_config.get("snowflake.schema")
    table = env_config.get("snowflake.inference_table")
    query = f"""
        SELECT COALESCE(MAX(PREDICTION_TIMESTAMP), '1900-12-31')::varchar::timestamp_ntz AS PREDICTION_TIMESTAMP
        FROM {db.upper()}.{schema.upper()}.{table.upper()}
    """
    return query
