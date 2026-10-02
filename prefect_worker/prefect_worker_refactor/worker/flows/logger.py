import sys

from prefect import get_run_logger


class Logger:
    def __init__(self, logger, level="info"):
        self.logger = getattr(logger, level)

    def write(self, message):
        if message != "\n":
            self.logger(message)

    def flush(self):
        pass


def setup_logger():
    """
    Adopt stdout and stderr to prefect logger
    """
    prefect_logger = get_run_logger()
    sys.stdout = Logger(prefect_logger, level="info")
    sys.stderr = Logger(prefect_logger, level="error")
    return prefect_logger
