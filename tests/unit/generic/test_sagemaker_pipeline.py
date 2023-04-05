import unittest

from mock import patch

from shapeshifter.generic.sagemaker_pipeline import SagemakerPipeline


class TestSagemakerPipeline(unittest.TestCase):
    def test_setup_abstract_class(self):
        with self.assertRaises(TypeError) as context:
            SagemakerPipeline()

    # @patch.multiple(SagemakerPipeline, __abstractmethods__=set())
    # def test_sagemaker_pipeline_class(self):
    #     SagemakerPipeline()
