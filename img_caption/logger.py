import logging
import sys


def get_logger(path_to_logfile):
    """
    Get logger
    :param path_to_logfile: path to logfile
    :return: Logger
    """

    logger = logging.getLogger("image_captioning")
    logger.setLevel(logging.INFO)

    # create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path_to_logfile)
    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # create formatters and add it to handlers
    stream_format = logging.Formatter("%(levelname)s - %(message)s")
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)
    file_handler.setFormatter(file_format)

    # add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def close_logger(logger):
    """
    Close logger
    Source: https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
    :param logger: logger
    """

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
