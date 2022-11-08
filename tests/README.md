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

First start up the local minio container, if it is not already running.  Then call `act -j pytest` to run the test suite.

```
# Start minio container in background
docker-compose -f tests/docker-compose.yml up -d

# Run local test suite
RUN_PRIVATE=1 act -j pytest
```
