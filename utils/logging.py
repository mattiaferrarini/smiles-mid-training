import logging

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def setup_logging(level=logging.INFO, fmt=DEFAULT_FORMAT):
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(level=level, format=fmt)

def get_logger(name=None):
    return logging.getLogger(name)
