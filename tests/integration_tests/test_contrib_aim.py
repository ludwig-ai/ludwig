import logging
import os
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "run_train_aim.py")


@pytest.mark.skip(reason="Aim integration not compatible with Aim 4.0.")
@pytest.mark.distributed
def test_contrib_experiment(csv_filename, tmpdir):
    aim_test_path = os.path.join(tmpdir, "results")
    os.makedirs(aim_test_path, exist_ok=True)

    os.environ["AIM_TEST_PATH"] = aim_test_path
    subprocess.call(["chmod", "-R", "a+w", os.environ["AIM_TEST_PATH"]])

    cmdline = [sys.executable, TEST_SCRIPT, "--csv-filename", csv_filename]
    print(cmdline)
    exit_code = subprocess.call(" ".join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0
