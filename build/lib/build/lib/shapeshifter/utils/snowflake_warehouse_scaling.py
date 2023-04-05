"""
Code to scale a snowflake warehouse using the ADA API. 
Call the module from CMD using below command;
Example:  python -m shapeshifter.utils.snowflake_warehouse_scaling.scale_snowflake_warehouse --size "LARGE"
"""

import argparse
import logging

import requests
from api_utils.api_requests import ApiRequests
from cerberus_utils.cerberus_reader import CerberusConfig

from ..config.env_config import EnvConfig


def scale_snowflake_warehouse(
    size: str,
    max_cluster_count: int = 1,
    change_reason: str = "Need to (re)scale for scheduled workload",
    ada_url="https://snowflake-api.h2o-prod.nikecloud.com",
):
    """Function to prepare the payload and make http request to ADA api to scale snowflake
       Confluence page link: https://confluence.nike.com/display/EMEATECH/Snowflake+API

    Args:
        size (str): warehouse size to be scaled to.
        max_cluster_count (int, optional): number of clusters. Defaults to 1.
        change_reason (str, optional): Defaults to "Need to (re)scale for scheduled workload".
        ada_url (str, optional): Defaults to "https://snowflake-api.h2o-prod.nikecloud.com".

    Returns:
        response of api
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(f"The scale_snowflake_warehouse with size: {size}")

    env_config = EnvConfig()
    cerberus_config = CerberusConfig(
        token=env_config.get("token"),
        user_key=env_config.get("cerberus.okta_user_key"),
        password_key=env_config.get("cerberus.okta_pwd_key"),
    )
    api_requests = ApiRequests.from_cerberus(
        cerberus_config,
        cerberus_sdb=env_config.get("cerberus.sdb_path_client"),
        okta_application_url=env_config.get("okta.application_url"),
    )

    payload = {
        "change_reason": change_reason,
        "max_cluster_count": max_cluster_count,
        "size": size,
    }
    warehouse_name = env_config.get("snowflake.read_warehouse").upper()

    try:
        response = api_requests.request(
            f"{ada_url}/v2/warehouses/{warehouse_name}",
            method=requests.put,
            data=payload,
        )
        response.raise_for_status()
        # check the warehouse size status
        warehouse_status_resp = api_requests.request(
            url=f"{ada_url}/v3/warehouses/{warehouse_name}",
            method=requests.get,
        )
        curr_size = warehouse_status_resp.json()["size"]
        logger.info(
            f"Snowflake has been scaled to {curr_size} with the response: {response.text}"
        )
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error: {response.reason}")
        logger.error(
            str(response.status_code) + ":" + f"when snowflake scaling to {size}"
        )
        raise err
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="Warehouse size, e.g; LARGE, XLARGE")
    args = parser.parse_args()
    scale_snowflake_warehouse(size=args.size)
