import contextlib
import random
import time


@contextlib.contextmanager
def retry_with_backoff(f_name: str, retries: int = 5, backoff_in_seconds: int = 1, logger=None):
    """Context manager that retries a function with exponential backoff.

    Source: https://keestalkstech.com/2021/03/python-utility-function-retry-with-exponential-backoff/

    Args:
        func_name: Name of the function to retry.
        retries: Number of times to retry the function.
        backoff_in_seconds: Initial backoff in seconds.
        logger: Logger to use for logging.
    """
    x = 0
    while True:
        try:
            yield
            break
        except Exception as e:
            if x == retries:
                raise e
            if logger:
                logger.debug(f"Retrying {f_name} due to {e}")
            sleep = backoff_in_seconds * 2**x + random.uniform(0, 1)
            time.sleep(sleep)
            x += 1


def retry(retries: int = 5, backoff_in_seconds: int = 1, logger=None):
    """Decorator that retries a function with exponential backoff.

    Source: https://keestalkstech.com/2021/03/python-utility-function-retry-with-exponential-backoff/

    Args:
        retries: Number of times to retry the function.
        backoff_in_seconds: Initial backoff in seconds.
    """

    def retry_with_backoff(f):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e

                    if logger:
                        logger.debug(f"Retrying {f.__name__} due to {e}")

                    sleep = backoff_in_seconds * 2**x + random.uniform(0, 1)
                    time.sleep(sleep)
                    x += 1

        return wrapper

    return retry_with_backoff
