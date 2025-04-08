"""Basic configurations for tiny-mirror."""

import logging


def get_logger() -> logging.Logger:
    """Gets the tiny-mirror logger."""
    logger = logging.getLogger("tiny-mirror")
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
