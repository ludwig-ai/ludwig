import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "run_train_wandb.py")


def test_contrib_experiment(csv_filename, tmpdir):
    wandb_dir = os.path.join(tmpdir, "results")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    subprocess.call(["chmod", "-R", "a+w", os.environ["WANDB_DIR"]])
    cmdline = [sys.executable, TEST_SCRIPT, "--csv-filename", csv_filename]
    exit_code = subprocess.call(" ".join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0


if __name__ == "__main__":
    """To run tests individually, run:

    ```python -m pytest tests/integration_tests/test_contrib_wandb.py::test_name```
    """
    pass
