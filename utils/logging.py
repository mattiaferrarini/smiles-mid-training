import logging

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level=logging.INFO, fmt=DEFAULT_FORMAT):
    """
    Sets up logging configuration

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG, etc...)
        fmt (str): Format string for log messages
    """
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(level=level, format=fmt)


def get_logger(name=None):
    """
    Retrieves a logger with the specified name

    Args:
        name (str, optional): Name of the logger. If None, the root logger is returned
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
