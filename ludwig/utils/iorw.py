# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import os
import fnmatch
import warnings
import entrypoints

from tenacity import retry, retry_if_exception_type, stop_after_attempt, \
    wait_exponential

from .exceptions import (
    LudwigException,
    LudwigRateLimitException,
    missing_dependency_generator
)

try:
    from gcsfs import GCSFileSystem
except ImportError:
    GCSFileSystem = missing_dependency_generator("gcsfs", "gcs")

# Handle newer and older gcsfs versions
try:
    try:
        from gcsfs.utils import HttpError as GCSHttpError
    except ImportError:
        from gcsfs.utils import HtmlError as GCSHttpError
except ImportError:
    # Fall back to a sane import if gcsfs is missing
    GCSHttpError = Exception

try:
    from gcsfs.utils import RateLimitException as GCSRateLimitException
except ImportError:
    # Fall back to GCSHttpError when using older library
    GCSRateLimitException = GCSHttpError

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class LudwigIO(object):
    '''
    The holder which houses any io system registered with the system.
    This object is used in a singleton manner to save and load particular
    named Handler objects for reference externally.
    '''

    def __init__(self):
        self.reset()

    def read(self, path, extensions=['.csv', '.hdf5', '.json']):
        if not fnmatch.fnmatch(os.path.basename(path), '*.*'):
            warnings.warn(
                "the file is not specified with any extension : " + os.path.basename(path)
            )
        elif not any(fnmatch.fnmatch(os.path.basename(path), '*' + ext) for ext in extensions):
            warnings.warn(
                "The specified input file ({}) does not end in one of {}".format(path, extensions)
            )
        file_metadata = self.get_handler(path).read(path)
        if isinstance(file_metadata, (bytes, bytearray)):
            return file_metadata.decode('utf-8')
        return file_metadata

    def write(self, buf, path, extensions=['.csv', '.hdf5', '.json']):
        # Usually no return object here
        if not fnmatch.fnmatch(os.path.basename(path), '*.*'):
            warnings.warn(
                "the file is not specified with any extension : " + os.path.basename(path)
            )
        elif not any(fnmatch.fnmatch(os.path.basename(path), '*' + ext) for ext in extensions):
            warnings.warn(
                "The specified input file ({}) does not end in one of {}".format(path, extensions)
            )
        return self.get_handler(path).write(buf, path)

    def exists(self, path):
        return self.get_handler(path).exists(path)

    def listdir(self, path):
        return self.get_handler(path).listdir(path)

    def pretty_path(self, path):
        return self.get_handler(path).pretty_path(path)

    def reset(self):
        self._handlers = []

    def register(self, scheme, handler):
        # Keep these ordered as LIFO
        self._handlers.insert(0, (scheme, handler))

    def register_entry_points(self):
        # Load handlers provided by other packages
        for entrypoint in entrypoints.get_group_all("ludwig.io"):
            self.register(entrypoint.name, entrypoint.load())

    def get_handler(self, path):
        local_handler = None
        for scheme, handler in self._handlers:
            if scheme == 'local':
                local_handler = handler

            if path.startswith(scheme):
                return handler

        if local_handler is None:
            raise LudwigException(
                "Could not find a registered schema handler for: {}".format(path)
            )

        return local_handler

    def get_handlers(self):
        return self._handlers


class GCSHandler(object):
    RATE_LIMIT_RETRIES = 3
    RETRY_DELAY = 1
    RETRY_MULTIPLIER = 1
    RETRY_MAX_DELAY = 4

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = GCSFileSystem()
        return self._client

    def read(self, path):
        with self._get_client().open(path) as f:
            return f.read()

    def exists(self, path):
        return self._get_client().exists(path)

    def listdir(self, path):
        return self._get_client().ls(path)

    def write(self, buf, path):
        # Wrapped so we can mock retry options during testing
        @retry(
            retry=retry_if_exception_type(LudwigRateLimitException),
            stop=stop_after_attempt(self.RATE_LIMIT_RETRIES),
            wait=wait_exponential(
                multiplier=self.RETRY_MULTIPLIER, min=self.RETRY_DELAY,
                max=self.RETRY_MAX_DELAY
            ),
            reraise=True,
        )
        def retry_write():
            try:
                with self._get_client().open(path, 'w') as f:
                    return f.write(buf)
            except (GCSHttpError, GCSRateLimitException) as e:
                try:
                    # If code is assigned but unknown, optimistically retry
                    if e.code is None or e.code == 429:
                        raise LudwigRateLimitException(e.message)
                except AttributeError:
                    try:
                        message = e.message
                    except AttributeError:
                        message = "Generic exception {} raised, " \
                                  "retrying".format(
                            type(e))
                    raise LudwigRateLimitException(message)
                # Reraise the original exception without retries
                raise

        return retry_write()

    def pretty_path(self, path):
        return path


# Instantiate a LudwigIO instance and register Handlers.
ludwig_io = LudwigIO()
ludwig_io.register("gs://", GCSHandler())
ludwig_io.register_entry_points()


def file_exists(path):
    """Verify if file exists locally or in remote location.

    :param path:
    :return:
    """
    for scheme, _ in ludwig_io.get_handlers():
        if path.startswith(scheme):
            logging.info('File exists in %s', scheme)
            return ludwig_io.exists(path)
    # Local file
    return os.path.isfile(path)
