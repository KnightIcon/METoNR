import logging


def get_logger():
    log_format = '[%(levelname)s] %(asctime)s  %(filename)s line %(lineno)d: %(message)s'
    date_fmt = '%a, %d %b %Y %H:%M:%S'
    logging.basicConfig(
        format=log_format,
        datefmt=date_fmt,
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger
