import os
import unittest

from pyspark.sql.types import StringType, StructField, StructType

from shapeshifter.pipelines.train.input_data_pipeline import TrainInputDataPipeline
from shapeshifter.utils.spark_helper import get_or_create_spark_session

# from ..base import S3Test


class TestDataPipeline(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls, *args):
    #     # super().setUpClass()
    #     cls.spark = get_or_create_spark_session()
    #     cls.dir_name = os.path.dirname(__file__)
    #     # cls.data_pipeline = InputDataPipeline()
    #     cls.start_date = "2022-11-20"
    #     cls.end_date = "2022-11-30"
    #     cls.input_schema = StructType(
    #         [
    #             StructField("sales_order_header_number", StringType(), True),
    #             StructField("sales_order_item_number", StringType(), True),
    #             StructField("sales_order_schedule_line_number", StringType(), True),
    #             StructField("product_code", StringType(), True),
    #         ]
    #     )
    #     cls.input_df = cls.spark.createDataFrame(
    #         [
    #             ("soh1", "soi1", "sosl1", "prod1"),
    #             ("soh2", "soi2", "sosl2", "prod2"),
    #             ("soh3", "soi3", "sosl3", "prod3"),
    #         ],
    #         schema=cls.input_schema,
    #     )
    #     cls.partition_by_write_columns = [
    #         "sales_order_header_number",
    #         "sales_order_item_number",
    #         "sales_order_schedule_line_number",
    #     ]

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    # def test_cannot_instantiate_concrete_classes_if_abstract_method_are_not_implemented(
    #     self,
    # ):
    #     class MyConcreteClassWithoutImplementations(InputDataPipeline):
    #         pass

    #     self.assertRaises(
    #         TypeError,
    #         lambda: MyConcreteClassWithoutImplementations(),
    #     )

    # def test_save_df_s3(self):
    #     self.input_df.write.partitionBy(self.partition_by_write_columns).format(
    #         "parquet"
    #     ).mode("overwrite").save(self.data_pipeline._DataPipeline__s3_output_folder)
    #     expected_output = self.spark.read.parquet(
    #         self.data_pipeline._DataPipeline__s3_output_folder
    #     )
    #     original_output = self.data_pipeline.save_df_s3(self.input_df)
    #     assert_pyspark_df_equal(original_output, expected_output)

    # @patch("shapeshifter.generic.data_pipeline.spark_helper.spark_sf_query")
    # def test_run(self, _spark_sf_query):
    #     _spark_sf_query.return_value = self.input_df
    #     original_output = self.data_pipeline.run(self.start_date, self.end_date)

    #     self.input_df.write.partitionBy(self.partition_by_write_columns).format(
    #         "parquet"
    #     ).mode("overwrite").save(self.data_pipeline._DataPipeline__s3_output_folder)
    #     expected_output = self.spark.read.parquet(
    #         self.data_pipeline._DataPipeline__s3_output_folder
    #     )
    #     assert_pyspark_df_equal(original_output, expected_output)


if __name__ == "__main__":
    unittest.main()
