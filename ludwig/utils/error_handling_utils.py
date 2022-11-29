import logging
from functools import partial

from retry.api import retry, retry_call

import ludwig.constants as const

logger = logging.getLogger(__name__)


default_retry_call = partial(
    retry_call, tries=const.TRIES, backoff=const.BACKOFF, delay=const.DELAY, jitter=const.JITTER, logger=logger
)


default_retry = partial(
    retry, tries=const.TRIES, backoff=const.BACKOFF, delay=const.DELAY, jitter=const.JITTER, logger=logger
)
