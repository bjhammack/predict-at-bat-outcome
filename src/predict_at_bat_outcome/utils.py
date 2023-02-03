from datetime import datetime
import functools
from glob import glob
import logging
from time import perf_counter


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        logging.info(f'{func.__name__!r} finished: {elapsed_time:0.4f}s')
        return value
    return wrapper_timer


def set_log_file(folder: str, prefix: str = 'log') -> str:
    now = datetime.now()
    filename = (
        f'{prefix}_{now.year}-{now.month}-{now.day}_'
        f'{now.hour}.{now.minute}.{now.second}.log'
        )
    return f'{folder}/{filename}'
