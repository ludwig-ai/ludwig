import pytest
from ludwig.benchmarking.profiler import LudwigProfiler


class MyPlugin:
    def pytest_sessionfinish(self):
        print("*** test run reporting finishing")


if __name__ == "__main__":
    with LudwigProfiler(tag="ludwig_pytest", output_dir="downloads/pytest_profiling", use_torch_profiler=False):
        pytest.main(["tests/ludwig/utils/test_metric_utils.py"], plugins=[MyPlugin()])
