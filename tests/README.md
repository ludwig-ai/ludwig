# Test Guide

Assuming your CWD is the Ludwig repo root.

## Basic

```bash
pytest -vs tests
```

## Private Tests

These tests connect to services like remote filesystems (Minio / S3), which can be run locally using Docker.

```bash
# prepare test services
docker-compose -f tests/docker-compose.yml up

# run all tests
RUN_PRIVATE=1 pytest -vs tests
```

## Slow Tests

These tests are very slow, and should typically be run on GPU machines.

```bash
RUN_SLOW=1 pytest -vs tests
```

## Running GitHub Actions Locally

It is possible to run the CI test suite locally by executing the `pytest` action using
[act](https://github.com/nektos/act).

First start up the local minio container, if it is not already running. Then call `act -j pytest` to run the test suite.

```
# Start minio container in background
docker-compose -f tests/docker-compose.yml up -d

# Run local test suite
RUN_PRIVATE=1 act -j pytest
```

## Tests that use ray clusters

Use the distributed pytest decorator to make sure that the test runs on CI jobs with the right ray dependencies installed.

```python
@pytest.mark.distributed
def test_something(ray_cluster_2_cpu):
    pass
```

Use module-level pytest fixtures to share ray cluster startup and teardown overhead at the module level. List of fixtures are found in `conftest.py`, for example:

```python
@pytest.fixture(scope="module")
def ray_cluster_2cpu(request):
    with _ray_start(request, num_cpus=2):
        yield
```

## Grouped Integration Tests

To leverage more runners to cut Ludwig CI time down, we partition `tests/integration_tests` into 3 groups (A, B, default). Each group should take on a roughly equal share of testing time, which at the time of writing is ~45 minutes each.

To define a new group and use it in tests:

1. Define a new pytest marker in `pytest.ini`.

```ini
integration_tests_a: mark a test to be run as part of integration tests, group A.
integration_tests_b: mark a test to be run as part of integration tests, group B.
# (new)
integration_tests_c: mark a test to be run as part of integration tests, group C.
```

2. Use the marker in a test file under `tests/integration_tests/`.

```python
import pytest

pytestmark = pytest.mark.integration_tests_c
```

If there's already a `pytestmark` declaration, turn it into a list.

```python
import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.integration_tests_c]
```

If there's a specific test to include in the group, decorate the test function.

```python
@pytest.mark.integration_tests_c
def test_something():
    pass
```

3. Create a new GHA to run pytest with that marker.

You can use [this change](https://github.com/ludwig-ai/ludwig/pull/3391/files#diff-2500680f4bc6c1b75c3d4b36372bf4d64c5f603b90bfd7a5186f66a20329d16aR189-R245) as a reference.

NOTE: Be sure to update other Integration Test GHA pytest jobs to exclude tests under the new marker.

To check which tests would be run under the `pytest` command without actually running them, use `--collect-only`.

```sh
pytest -m "not distributed and not slow and not combinatorial and not llm and integration_tests_c" --junitxml pytest.xml tests/integration_tests --collect-only
```
