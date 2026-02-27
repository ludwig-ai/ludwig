import importlib.util
import logging
import os
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "run_train_comet.py")


@pytest.mark.skipif(
    not importlib.util.find_spec("pkg_resources"),
    reason="comet_ml requires pkg_resources (removed in setuptools 82+)",
)
def test_contrib_experiment(csv_filename):
    cmdline = [sys.executable, TEST_SCRIPT, "--csv-filename", csv_filename]
    exit_code = subprocess.call(" ".join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0


if __name__ == "__main__":
    """To run tests individually, run:

    ```python -m pytest tests/integration_tests/test_contrib_comet.py::test_name```
    """
