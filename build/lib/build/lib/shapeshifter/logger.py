"""
`logger` module contains a
[logger object](https://docs.python.org/3/library/logging.html#logger-objects)
configured with [Rich logging handler](https://rich.readthedocs.io/en/stable/logging.html)
that is intended to be used as a singleton across the application code instead
of using print statements.

!!! Example
    ```py title="hello_world.py" linenums="1"
    from src.logger import logger

    logger.info("Hello world!")
    ```

    ```console  title="terminal"
    >>> python hello_world.py
    [11:01:44] INFO     Hello world!                                    hello_world.py:3
    ```
"""

import logging

from rich.logging import RichHandler

from shapeshifter.enums import LoggingLevel


def get_logger(
    name: str,
    level: LoggingLevel = LoggingLevel.INFO,
) -> logging.Logger:
    """Get a logger configured with Rich logging handler.

    Args:
        name: Name to assign to the logger.
        level: Lowest logging level.

    Returns:
        A logger instance.
    """

    # Configure basic log level to ERROR and add Rich handler
    logging.basicConfig(
        level="ERROR",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            ),
        ],
    )

    # Initialize logger and configure logging level
    rich_logger = logging.getLogger(name=name)
    rich_logger.setLevel(level.value)

    return rich_logger


logger = get_logger(
    name="shapeshifter",
)
