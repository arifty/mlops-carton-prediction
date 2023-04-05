"""
`enums` module implements definitions of enumerated variables.
"""

import enum


@enum.unique
class LoggingLevel(str, enum.Enum):
    """Enumerates possible logging levels.

    !!! Info "Members"

        - CRITICAL
        - ERROR
        - WARNING
        - INFO
        - DEBUG
        - NOTSET
    """

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"
