import pytest

from ludwig.constants import TRIES
from ludwig.utils.error_handling_utils import default_retry


def test_default_retry_success():
    ctr = 0

    @default_retry()
    def flaky_function():
        nonlocal ctr
        if ctr < TRIES - 1:
            ctr += 1
            raise Exception(f"Ctr: {ctr} too low.")

        return

    flaky_function()


def test_default_retry_failure():
    ctr = 0

    @default_retry()
    def flaky_function():
        nonlocal ctr
        if ctr < TRIES:
            ctr += 1
            raise Exception(f"Ctr: {ctr} too low.")

        return

    with pytest.raises(Exception):
        flaky_function()


def test_default_retry_success_custom_num_tries():
    CUSTOM_TRIES = 3
    ctr = 0

    @default_retry(tries=CUSTOM_TRIES)
    def flaky_function():
        nonlocal ctr
        if ctr < CUSTOM_TRIES - 1:
            ctr += 1
            raise Exception(f"Ctr: {ctr} too low.")

        return

    flaky_function()


def test_default_retry_does_not_retry_non_oserror():
    """Non-OSError exceptions (e.g. AttributeError from a broken import) must propagate immediately.

    Regression test for https://github.com/ludwig-ai/ludwig/issues/4170: torchao raises
    AttributeError on PyTorch < 2.7 and the retry wrapper was retrying 8 times, wasting
    ~2.5 minutes before surfacing the real error.
    """
    call_count = 0

    @default_retry(tries=8, exceptions=OSError)
    def always_attribute_error():
        nonlocal call_count
        call_count += 1
        raise AttributeError("module 'torch.utils._pytree' has no attribute 'register_constant'")

    with pytest.raises(AttributeError):
        always_attribute_error()

    assert call_count == 1, f"Expected 1 call (no retries), got {call_count}"


def test_default_retry_retries_oserror():
    """OSError (transient network/IO) should still be retried."""
    call_count = 0
    CUSTOM_TRIES = 3

    @default_retry(tries=CUSTOM_TRIES, exceptions=OSError)
    def flaky_network():
        nonlocal call_count
        call_count += 1
        if call_count < CUSTOM_TRIES:
            raise OSError("connection reset by peer")

    flaky_network()
    assert call_count == CUSTOM_TRIES
