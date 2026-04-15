# Anomaly Detection with Deep SVDD, SAD, and DROCC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/anomaly_detection/anomaly_detection.ipynb)

This example shows how to train an anomaly detection model with Ludwig using the `anomaly` output feature type. The model learns a compact representation of "normal" sensor data using three complementary hypersphere-based objectives:

- **Deep SVDD** — unsupervised, trains only on normal samples
- **Deep SAD** — semi-supervised, uses a small set of labeled anomalies at training time
- **DROCC** — unsupervised with adversarial robustness, recommended for expressive encoders

At inference time each sample receives an `anomaly_score` equal to its squared distance from the learned hypersphere centre. Higher scores indicate more anomalous samples.

## Prerequisites

```bash
pip install ludwig
```

## Dataset

The example uses a synthetic sensor dataset with four numeric features (`sensor_a`, `sensor_b`, `sensor_c`, `timestamp_hour`). Normal samples are drawn from a Gaussian distribution centred at the origin; anomalous samples have a large offset. The train split contains **only normal samples**; the test split contains both normal and anomalous samples for evaluation.

## Loss variants

### Deep SVDD (unsupervised)

```yaml
output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: deep_svdd
      nu: 0.1   # fraction of points allowed outside the hypersphere
```

Hard-boundary objective: minimise the mean squared distance of all normal training representations to the hypersphere centre `c`. The `nu` parameter controls soft-boundary relaxation (set to `0` for hard SVDD).

Full config: [`config_deep_svdd.yaml`](config_deep_svdd.yaml)

### Deep SAD (semi-supervised)

```yaml
output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: deep_sad
      eta: 1.0   # weight for the labeled anomaly repulsion term
```

Extends Deep SVDD with labeled anomaly support. Normal and unlabeled samples (label `0` or `-1`) are pulled toward `c`; labeled anomalies (label `1`) are pushed away. Provide a small fraction of labeled anomaly rows in the training data with `anomaly=1`.

Full config: [`config_deep_sad.yaml`](config_deep_sad.yaml)

### DROCC (robust unsupervised)

```yaml
output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: drocc
      perturbation_strength: 0.1
      num_perturbation_steps: 5
```

Prevents hypersphere collapse via an adversarial perturbation regulariser. Recommended when using expressive encoders (e.g. transformers) that are prone to degenerate solutions where all representations collapse to a single point.

Full config: [`config_drocc.yaml`](config_drocc.yaml)

## Running the example

### CLI

```bash
# Train
ludwig train --config config_deep_svdd.yaml --dataset /tmp/sensors_train.csv

# Predict (score test samples)
ludwig predict --model_path results/experiment_run/model \
               --dataset /tmp/sensors_test.csv

# Evaluate (requires labeled anomaly column in test CSV)
ludwig evaluate --model_path results/experiment_run/model \
                --dataset /tmp/sensors_test.csv
```

### Python API

```python
import pandas as pd
from ludwig.api import LudwigModel

# Load data
train_df = pd.read_csv("/tmp/sensors_train.csv")
test_df = pd.read_csv("/tmp/sensors_test.csv")

# Train
model = LudwigModel("config_deep_svdd.yaml", logging_level="ERROR")
results = model.train(dataset=train_df)

# Predict — returns a DataFrame with anomaly_score_predictions column
predictions, _ = model.predict(dataset=test_df)
print(predictions[["anomaly_anomaly_score_predictions"]].describe())
```

For a full walkthrough including score distribution plots and AUC comparison, open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/anomaly_detection/anomaly_detection.ipynb)

## Files

| File                      | Description                               |
| ------------------------- | ----------------------------------------- |
| `anomaly_detection.ipynb` | End-to-end Colab notebook                 |
| `config_deep_svdd.yaml`   | Deep SVDD config                          |
| `config_deep_sad.yaml`    | Deep SAD (semi-supervised) config         |
| `config_drocc.yaml`       | DROCC config                              |
| `train.py`                | Standalone training and evaluation script |
