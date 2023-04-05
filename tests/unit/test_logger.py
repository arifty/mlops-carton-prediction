import logging

from shapeshifter.enums import LoggingLevel
from shapeshifter.logger import get_logger


def test_name() -> None:
    name = "test"
    assert get_logger(name=name).name == name


def test_level(caplog) -> None:
    logger = get_logger(name="Test", level=LoggingLevel.ERROR)

    msg = "hello world"

    with caplog.at_level(logging.INFO):
        logger.info(msg)
        assert msg not in caplog.text
        logger.error(msg)
        assert msg in caplog.text
