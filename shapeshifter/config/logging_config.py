"""
Logging configuration
"""
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s "
            "LOGGER_NAME:%(name)s "
            "LEVEL_NAME:%(levelname)-4s "
            "FILE_NAME:%(module)s/%(filename)s - %(funcName)s:%(lineno)d "
            "MESSAGE:%(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
