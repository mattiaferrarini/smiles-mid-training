import logging
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def setup_logging(level: int = logging.INFO, fmt: str = DEFAULT_FORMAT) -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(level=level, format=fmt)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
