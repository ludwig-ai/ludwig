import argparse
from collections.abc import Sequence

import pytest

from ludwig.contrib import add_contrib_callback_args
from ludwig.contribs.aim import AimCallback
from ludwig.contribs.comet import CometCallback
from ludwig.contribs.mlflow import MlflowCallback
from ludwig.contribs.wandb import WandbCallback


@pytest.mark.parametrize(
    "sys_argv,expected",
    [
        ([], []),
        (["--mlflow"], [MlflowCallback]),
        (["--aim"], [AimCallback]),
        (["--comet"], [CometCallback]),
        (["--wandb"], [WandbCallback]),
    ],
)
def test_add_contrib_callback_args(sys_argv: Sequence[str], expected: list[type]):
    parser = argparse.ArgumentParser()
    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)
    callbacks = args.callbacks or []

    assert len(callbacks) == len(expected)
    for callback, expected_cls in zip(callbacks, expected):
        assert isinstance(callback, expected_cls)
