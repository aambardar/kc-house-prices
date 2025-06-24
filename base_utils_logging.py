import logging
import inspect
import os
from proj_configs import LOG_FILE, PATH_OUT_LOGS, LOG_ROOT_LEVEL, LOG_FILE_LEVEL, LOG_CONSOLE_LEVEL

class CustomLogger:

    def __init__(self):
        self.logger = logging.getLogger()

        BASE_FORMAT = "[%(name)s][%(levelname)-8s]%(message)s"
        FILE_FORMAT = "[%(asctime)s]" + BASE_FORMAT

        os.makedirs(os.path.dirname(PATH_OUT_LOGS), exist_ok=True)
        self.logger.setLevel(LOG_ROOT_LEVEL)

        file_logger = logging.FileHandler(LOG_FILE)
        file_logger.setLevel(LOG_FILE_LEVEL)
        file_logger.setFormatter(logging.Formatter(FILE_FORMAT))
        self.logger.addHandler(file_logger)

        console_logger = logging.StreamHandler()
        console_logger.setLevel(LOG_CONSOLE_LEVEL)
        # In this case, the BASE_FORMAT was used, because it’s going to the terminal, where timestamps can cause excessive noise.
        console_logger.setFormatter(logging.Formatter(BASE_FORMAT))
        self.logger.addHandler(console_logger)

    def debug(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.debug(f'[{log_invoker}] {message}')

    def info(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.info(f'[{log_invoker}] {message}')

    def warning(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.warning(f'[{log_invoker}] {message}')

    def error(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.error(f'[{log_invoker}] {message}')

    def critical(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.critical(f'[{log_invoker}] {message}')


def setup_logging():
    global logger
    logger = CustomLogger()