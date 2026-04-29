# Deploying Ludwig Models with KServe

[KServe](https://kserve.github.io/website/) is the standard Kubernetes serving runtime
for ML models. Ludwig ships `ludwig.serve_kserve` which wraps any trained `LudwigModel`
behind the KServe **Open Inference Protocol v2** (`/v2/models/{name}/infer`) so Ludwig
models slot into existing MLOps pipelines that expect v2-compliant endpoints.

## Local testing (no Kubernetes required)

```bash
pip install "ludwig[serve]" kserve

# Start the server
python -m ludwig.serve_kserve \
  --model_name titanic \
  --model_path ./results/experiment_run/model \
  --http_port 8080

# Predict with v2 protocol
curl -s -X POST http://localhost:8080/v2/models/titanic/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "Pclass", "shape": [2], "datatype": "INT64", "data": [1, 3]},
      {"name": "Sex",    "shape": [2], "datatype": "BYTES",  "data": ["female", "male"]},
      {"name": "Age",    "shape": [2], "datatype": "FP32",   "data": [28.0, 22.0]}
    ]
  }'
```

## Kubernetes deployment

1. **Build and push the Ludwig image** (or use the public one):

```bash
docker build -t your-registry/ludwig:latest .
docker push your-registry/ludwig:latest
```

2. **Copy your trained model** to a `PersistentVolume` or an object-store URI.

1. **Apply the manifest**:

```bash
# Edit serving_config.yaml to point to your model and image, then:
kubectl apply -f serving_config.yaml
kubectl get inferenceservice ludwig-titanic
```

4. **Send predictions**:

```bash
INGRESS=$(kubectl get svc istio-ingressgateway -n istio-system \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl -s -H "Host: ludwig-titanic.default.example.com" \
  http://${INGRESS}/v2/models/titanic/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "Pclass", "shape": [1], "datatype": "INT64", "data": [1]},
      {"name": "Sex",    "shape": [1], "datatype": "BYTES",  "data": ["female"]},
      {"name": "Age",    "shape": [1], "datatype": "FP32",   "data": [28.0]}
    ]
  }'
```

## Programmatic usage

```python
from ludwig.serve_kserve import serve_ludwig_model

# Blocking — runs until Ctrl-C
serve_ludwig_model(
    model_name="titanic",
    model_path="./results/experiment_run/model",
    http_port=8080,
)
```

## v2 protocol reference

| field               | description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `inputs[].name`     | Ludwig input feature name                                                  |
| `inputs[].shape`    | `[batch_size]` (1-D flat batch)                                            |
| `inputs[].datatype` | `BYTES` for text/category, `FP32`/`FP64` for numbers, `INT64` for integers |
| `inputs[].data`     | Flat list of values, length == batch_size                                  |

Response `outputs` follow the same shape. All output values are currently serialised
as `BYTES` (string representation); numeric output feature types will be exposed as
`FP32`/`INT64` in a future release.
