# Ray Job Submission for Ludwig

Run Ludwig training on a remote Ray cluster using Ray Job Submission instead of Ray Client.

## Why Ray Job Submission?

Ray Client mode (`ray.init("ray://head:10001")`) has known issues with `ray.data` operations ([ray-project/ray#47759](https://github.com/ray-project/ray/issues/47759)), causing `OwnerDiedError` and similar failures during distributed training. Ray Job Submission avoids this entirely by running the training script directly on the cluster head node.

## How it works

```
Your machine                          Ray Cluster
+------------------+                  +------------------+
| submit_job.py    | --- uploads ---> | train_on_cluster.py
| config.yaml      |     config +     | (runs on head node)
|                  |     script       | ray.init() is local
|                  | <-- streams ---  | ludwig.train()
|                  |     logs back    | saves to S3/NFS
+------------------+                  +------------------+
```

1. `submit_job.py` runs on your machine and uploads the config + training script
1. `train_on_cluster.py` runs on the cluster head node
1. `ray.init()` connects locally (no Client mode)
1. Ludwig distributes training across workers normally
1. Model is saved to shared storage (S3/GCS/NFS)

## Prerequisites

**Your machine:**

```bash
pip install "ray[default]"
```

**Ray cluster:**

```bash
pip install "ludwig[distributed]"
```

Or install at job start with `--pip ludwig[distributed]` (adds cold start time).

## Usage

```bash
# Basic usage
python submit_job.py \
    --ray-address http://ray-head:8265 \
    --config config.yaml \
    --dataset s3://my-bucket/data/train.csv \
    --output-dir s3://my-bucket/results/

# KubeRay cluster
python submit_job.py \
    --ray-address http://ray-head.ray.svc:8265 \
    --config config.yaml \
    --dataset s3://my-bucket/data/train.csv \
    --output-dir s3://my-bucket/results/

# Install Ludwig on the fly (no pre-install on cluster)
python submit_job.py \
    --ray-address http://ray-head:8265 \
    --config config.yaml \
    --dataset s3://my-bucket/data/train.csv \
    --output-dir /shared/nfs/results/ \
    --pip "ludwig[distributed]"

# Submit without waiting for results
python submit_job.py \
    --ray-address http://ray-head:8265 \
    --config config.yaml \
    --dataset s3://my-bucket/data/train.csv \
    --output-dir s3://my-bucket/results/ \
    --no-follow
```

## Data access

The dataset must be accessible **from the cluster**, not from your machine:

| Storage | Example path               | Notes                         |
| ------- | -------------------------- | ----------------------------- |
| S3      | `s3://bucket/data.csv`     | Cluster needs AWS credentials |
| GCS     | `gs://bucket/data.csv`     | Cluster needs GCP credentials |
| NFS     | `/shared/data/train.csv`   | Must be mounted on all nodes  |
| HDFS    | `hdfs://namenode/data.csv` | Hadoop cluster                |

If your data is local, upload it first:

```bash
aws s3 cp my_data.csv s3://my-bucket/data/my_data.csv
```

## Files

- `submit_job.py` -- runs on your machine, submits the job
- `train_on_cluster.py` -- runs on the cluster, does the actual training
- `config.yaml` -- sample Ludwig config (customize for your task)

## Customization

**Using your own config**: Replace `config.yaml` with your Ludwig config. Any valid Ludwig config works.

**Requesting GPUs for the driver**: Use `--num-gpus 1` if your training script needs GPU access on the head node.

**Custom runtime environment**: Edit `submit_job.py` to add `runtime_env` options like `conda`, `container`, or `env_vars`.
