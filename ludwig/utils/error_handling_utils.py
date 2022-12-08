import logging
import random
import time
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional

import ludwig.constants as const

logger = logging.getLogger(__name__)


def _retry_internal(
    fn: Callable,
    tries: int = 5,
    backoff: int = 2,
    delay: int = 1,
    jitter=(0, 1),
    logger: Optional[logging.Logger] = None,
    exceptions: Optional[List[Any]] = [],
    predicate: Optional[Callable] = None,
):
    """Retry a function with exponential backoff.

    Args:
        fn: The function to retry.
        tries: The number of times to retry. This excludes the initial try, so a value of n means (n + 1) tries.
        backoff: The backoff multiplier. E.g. value of 2 will double the delay each retry.
        delay: The initial delay. This is the delay before the first retry.
        jitter: This is a tuple of (min, max) jitter values, where the jitter is randomly sampled from this range.
        logger: The logger to use.
        exceptions: A list of exceptions to retry on. If empty, retry on all exceptions.
        predicate: The predicate to retry on. If not None, then only retry when the predicate evaluates to True.
    """

    x = 0
    if exceptions and type(exceptions) != list:
        exceptions = [exceptions]
    while True:
        try:
            return fn()
        except Exception as e:
            if predicate and not predicate(e):
                raise e
            if exceptions and not any(isinstance(e, exception) for exception in exceptions):
                raise e
            if x == tries:
                raise e
            sleep = delay * backoff**x + random.uniform(*jitter)
            if logger is not None:
                logger.warning(
                    f"Encountered exception '{e}' in function '{fn.func.__name__}', retrying in {sleep} seconds..."
                )
            time.sleep(sleep)
            x += 1


def retry(
    tries: int = 5,
    backoff: int = 2,
    delay: int = 1,
    jitter=(0, 1),
    logger: Optional[logging.Logger] = None,
    exceptions: Optional[List[Any]] = [],
    predicate: Optional[Callable] = None,
):
    """Retry a function with exponential backoff.

    Args:
        fn: The function to retry.
        tries: The number of times to retry. This excludes the initial try, so a value of n means (n + 1) tries.
        backoff: The backoff multiplier. E.g. value of 2 will double the delay each retry.
        delay: The initial delay. This is the delay before the first retry.
        jitter: This is a tuple of (min, max) jitter values, where the jitter is randomly sampled from this range.
        logger: The logger to use.
        exceptions: A list of exceptions to retry on. If empty, retry on all exceptions.
        predicate: The predicate to retry on. If not None, then only retry when the predicate evaluates to True.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return _retry_internal(
                partial(fn, *args, **kwargs),
                tries=tries,
                backoff=backoff,
                delay=delay,
                jitter=jitter,
                logger=logger,
                exceptions=exceptions,
                predicate=predicate,
            )

        return wrapper

    return decorator


def retry_call(
    f: Callable,
    fargs: Optional[List] = None,
    fkwargs: Optional[Dict] = None,
    tries: int = 5,
    backoff: int = 2,
    delay: int = 1,
    jitter=(0, 1),
    logger: Optional[logging.Logger] = None,
    exceptions: Optional[List[Any]] = [],
    predicate: Optional[Callable] = None,
):
    """Retry a function with exponential backoff.

    Args:
    fn: The function to retry.
    tries: The number of times to retry. This excludes the initial try, so a value of n means (n + 1) tries.
    backoff: The backoff multiplier. E.g. value of 2 will double the delay each retry.
    delay: The initial delay. This is the delay before the first retry.
    jitter: This is a tuple of (min, max) jitter values, where the jitter is randomly sampled from this range.
    logger: The logger to use.
    exceptions: A list of exceptions to retry on. If empty, retry on all exceptions.
    predicate: The predicate to retry on. If not None, then only retry when the predicate evaluates to True.
    """
    args = fargs if fargs else list()
    kwargs = fkwargs if fkwargs else dict()
    return _retry_internal(
        partial(f, *args, **kwargs),
        tries=tries,
        backoff=backoff,
        delay=delay,
        jitter=jitter,
        logger=logger,
        exceptions=exceptions,
        predicate=predicate,
    )


default_retry_call = partial(
    retry_call, tries=const.TRIES, backoff=const.BACKOFF, delay=const.DELAY, jitter=const.JITTER, logger=logger
)


default_retry = partial(
    retry, tries=const.TRIES, backoff=const.BACKOFF, delay=const.DELAY, jitter=const.JITTER, logger=logger
)


def x_minio_backend_down_predicate(e: Exception) -> bool:
    from botocore.exceptions import ClientError

    if isinstance(e, ClientError):
        if e.response["Error"]["Code"] == "XMinioBackendDown":
            return True

    if isinstance(e, OSError):
        if "XMinioBackendDown" in e.strerror:
            return True

    return False


# This method adds infinite retry with exponential backoff for XMinioBackendDown errors.
x_minio_backend_down_retry = partial(
    retry,
    tries=-1,
    backoff=const.BACKOFF,
    delay=const.DELAY,
    jitter=const.JITTER,
    logger=logger,
    predicate=x_minio_backend_down_predicate,
)
