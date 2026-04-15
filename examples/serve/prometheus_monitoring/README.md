# Ludwig Serving + Prometheus + Grafana Monitoring

This directory contains a Docker Compose stack for running a Ludwig vLLM server alongside Prometheus and Grafana so you can monitor inference metrics in real time.

## Prerequisites

- Docker 24+ and Docker Compose v2
- NVIDIA Container Toolkit (for GPU support)
- A trained Ludwig LLM model directory on the host

## Quick start

1. **Set the model path** (the directory Ludwig will serve):

   ```bash
   export MODEL_PATH=/absolute/path/to/your/ludwig_model
   ```

1. **Start the stack**:

   ```bash
   docker compose up -d
   ```

1. **Check services are healthy**:

   ```bash
   docker compose ps
   ```

1. **Send a test prediction**:

   ```bash
   curl http://localhost:8000/predict -X POST -F 'text=Hello world'
   ```

## Services

| Service        | URL                   | Description                      |
| -------------- | --------------------- | -------------------------------- |
| `ludwig-serve` | http://localhost:8000 | Ludwig vLLM inference server     |
| `prometheus`   | http://localhost:9090 | Prometheus metrics scraper       |
| `grafana`      | http://localhost:3000 | Grafana dashboards (admin/admin) |

## Prometheus metrics

Ludwig exposes a `/metrics` endpoint in [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/).
Prometheus is configured to scrape it every 15 seconds (see `prometheus.yml`).

Key metrics to watch:

- `ludwig_request_latency_seconds` — per-request inference latency histogram
- `ludwig_requests_total` — total requests served (labelled by endpoint)
- `ludwig_batch_size` — distribution of batch sizes received by `/batch_predict`

## Grafana

Open http://localhost:3000 and log in with `admin` / `admin`.

The Prometheus datasource is pre-provisioned. To import a ready-made dashboard:

1. Click **Dashboards → Import**
1. Paste the dashboard ID or upload a JSON file
1. Select the **Prometheus** datasource

## Stopping the stack

```bash
docker compose down
```

To also remove persistent volumes (Prometheus TSDB and Grafana data):

```bash
docker compose down -v
```

## Configuration

- **`prometheus.yml`** — scrape interval and target configuration
- **`docker-compose.yml`** — service definitions, port mappings, GPU allocation
- Set `GF_SECURITY_ADMIN_PASSWORD` in `docker-compose.yml` before deploying to production
