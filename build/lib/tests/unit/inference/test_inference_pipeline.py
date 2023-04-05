# import json
# import os
# import unittest
# from unittest.mock import patch

# import boto3
# from aws_utils.s3_lib import S3Proxy
# from moto import mock_s3

# from shapeshifter.config.env_config import EnvConfig
# from shapeshifter.inference.ml_pipeline import InferenceMlPipeline


# # @mock_sagemaker
# @mock_s3
# # @patch(boto3.client("sagemaker", region_name="eu-west-1"))
# class TestInferenceMlPipeline(unittest.TestCase):
#     # @classmethod
#     def setUp(self, *args):
#         self.resource = boto3.resource("s3")
#         self.resource.create_bucket(
#             Bucket="local-bucket",
#             CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
#         )
#         self.s3_lib = S3Proxy("local-bucket")
#         self.env_config = EnvConfig()

#         train_config = {
#             "encoders_s3_path": "s3://local-bucket/shapeshifter/encoders",
#             "model_location": "s3://local-bucket/shapeshifter/model_location",
#         }

#         inference_config = {
#             "inference_data_csv": "s3://local-bucket/shapeshifter/inference_data.csv",
#             "inference_data_parquet": "s3://local-bucket/shapeshifter/parquet/",
#         }

#         self.file = os.path.join("/tmp/train_config.json")
#         with open("{}".format(self.file), "w") as f:
#             json.dump(train_config, f)

#         self.file = os.path.join("/tmp/inference_config.json")
#         with open("{}".format(self.file), "w") as f:
#             json.dump(inference_config, f)

#         self.s3_lib.upload_file(
#             "/tmp/train_config.json",
#             f"{self.env_config.get('s3_keys.train_config')}",
#         )

#         self.s3_lib.upload_file(
#             "/tmp/inference_config.json",
#             f"{self.env_config.get('s3_keys.inference_config')}",
#         )

#         self.mlpipeline = InferenceMlPipeline(self.s3_lib)

#     def test_trigger_sagemaker_pipeline(self):
#         self.mlpipeline.trigger_sagemaker_pipeline()
