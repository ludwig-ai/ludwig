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
