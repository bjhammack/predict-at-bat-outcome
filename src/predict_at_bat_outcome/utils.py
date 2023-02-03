from datetime import datetime
import functools
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


def set_dated_file(folder: str, prefix: str, suffix: str) -> str:
    def format_datetime(dt):
        year = str(dt.year)[-2:]
        month = str(dt.month) if dt.month > 9 else '0'+str(dt.month)
        day = str(dt.day) if dt.day > 9 else '0'+str(dt.day)
        hour = str(dt.hour) if dt.hour > 9 else '0'+str(dt.hour)
        minute = str(dt.minute) if dt.minute > 9 else '0'+str(dt.minute)
        second = str(dt.second) if dt.second > 9 else '0'+str(dt.second)
        return year, month, day, hour, minute, second

    now = datetime.now()
    year, month, day, hour, minute, second = format_datetime(now)
    filename = (
        f'{prefix}_{year}{month}{day}_{hour}{minute}{second}{suffix}'
        )
    return f'{folder}/{filename}'
