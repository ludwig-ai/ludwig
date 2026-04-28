# Deploying Ludwig Models with Ray Serve

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is a production-grade model
serving library built on Ray that supports autoscaling, traffic splitting, and rolling
updates. Ludwig ships `ludwig.serve_ray_serve` to wrap any trained `LudwigModel` as a
Ray Serve deployment with a single function call.

## Prerequisites

```bash
pip install "ludwig[distributed]"   # pulls in ray[serve]
```

## Quick start

1. **Train a model** (or use an existing one):

```bash
ludwig train \
  --config examples/titanic/simple_model_training.yaml \
  --dataset examples/titanic/titanic.csv \
  --output_directory ./results
```

2. **Deploy**:

```bash
python deploy.py --model_path ./results/experiment_run/model --block
```

3. **Predict**:

```bash
# Single record
curl -s -X POST http://localhost:8000/ludwig \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Sex": "female", "Age": 28}'

# Batch
curl -s -X POST http://localhost:8000/ludwig \
  -H "Content-Type: application/json" \
  -d '[{"Pclass": 3, "Sex": "male", "Age": 22}, {"Pclass": 1, "Sex": "female", "Age": 35}]'
```

## GPU deployment

```bash
python deploy.py \
  --model_path ./results/experiment_run/model \
  --num_replicas 2 \
  --gpu \
  --block
```

## Programmatic usage

```python
import ray
from ludwig.serve_ray_serve import deploy_ludwig_model

ray.init()

handle = deploy_ludwig_model(
    model_path="./results/experiment_run/model",
    name="titanic",
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
)

# Programmatic call (no HTTP)
import asyncio, pandas as pd

result = asyncio.get_event_loop().run_until_complete(handle.predict.remote({"Pclass": 1, "Sex": "female", "Age": 28}))
print(result)
```

## API contract

| endpoint  | method | body                      | response                                    |
| --------- | ------ | ------------------------- | ------------------------------------------- |
| `/{name}` | POST   | single JSON record (dict) | dict with one prediction per output feature |
| `/{name}` | POST   | list of JSON records      | `{"predictions": [...]}`                    |

The payload shape mirrors Ludwig's existing `ludwig.serve_v2` FastAPI server so clients
can switch backends without code changes.
