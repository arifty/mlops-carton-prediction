import unittest
from logging import Logger

from freezegun import freeze_time

from shapeshifter.utils.generic_functions import (
    camelcase_to_snake,
    get_logger,
    parse_arg_dates,
    snake_case_to_camel,
)


class TestGenericFunctions(unittest.TestCase):
    def test_camelcase_to_snake(self):
        self.assertEqual("a_b_c", camelcase_to_snake("ABC"))
        self.assertEqual("ab_c", camelcase_to_snake("AbC"))
        self.assertEqual("abc", camelcase_to_snake("abc"))
        self.assertEqual("ab_c", camelcase_to_snake("ab_c"))
        self.assertEqual("ab__c", camelcase_to_snake("ab_C"))

    def test_snake_case_to_camel(self):
        self.assertEqual("ABC", snake_case_to_camel("a_b_c"))
        self.assertEqual("AbC", snake_case_to_camel("ab_c"))
        self.assertEqual("Abc", snake_case_to_camel("abc"))
        self.assertEqual("AbC", snake_case_to_camel("ab_c"))
        self.assertEqual("AbC", snake_case_to_camel("ab__c"))

    @freeze_time("2023-03-16")
    def test_parse_arg_dates(self):
        t = parse_arg_dates("2022-01-01", "2022-02-01")
        self.assertEqual(t[0], "2022-01-01")
        self.assertEqual(t[1], "2022-02-01")

        t = parse_arg_dates("", "")
        self.assertEqual(t[0], "2021-03-16")
        self.assertEqual(t[1], "9999-12-31")

        with self.assertRaises(ValueError):
            parse_arg_dates("2023-01-01", "2022-02-01")

    def test_get_logger(self):
        l = get_logger(
            logging_dict={
                "version": 1,
            }
        )
        self.assertIsInstance(l, Logger)
        self.assertEqual(l.name, "shapeshifter.utils.generic_functions")

        l2 = get_logger(
            logging_dict={
                "version": 1,
            },
            logger_name="testingname",
        )
        self.assertIsInstance(l2, Logger)
        self.assertEqual(l2.name, "testingname")


if __name__ == "__main__":
    unittest.main()
