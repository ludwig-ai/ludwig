"""Submit a Ludwig training job to a remote Ray cluster.

This script runs on YOUR machine (laptop, CI server, etc.) and submits the
training to run on the Ray cluster. It avoids Ray Client mode entirely,
which has known issues with ray.data (ray-project/ray#47759).

Usage:
    python submit_job.py \
        --ray-address http://ray-head:8265 \
        --config config.yaml \
        --dataset s3://my-bucket/data/train.csv \
        --output-dir s3://my-bucket/results/

Requirements:
    pip install ray[default]  # for JobSubmissionClient
    The Ray cluster must have ludwig[distributed] installed, OR you can
    specify it in --pip-packages so it's installed at job start (slower).

How it works:
    1. Uploads config.yaml + train_on_cluster.py to the cluster via runtime_env
    2. Submits a Ray Job that runs train_on_cluster.py on the head node
    3. Streams logs back to your terminal
    4. The trained model is saved to --output-dir (use S3/GCS for easy retrieval)

Data handling:
    The dataset must be accessible FROM the cluster, not from your machine.
    Options:
      - S3/GCS: pass an s3:// or gs:// URI. Ludwig uses fsspec to read it.
      - NFS: if your cluster has a shared filesystem, pass the NFS path.
      - HDFS: pass an hdfs:// URI.
    Local files on your machine won't work unless you upload them first.

    If you need to upload a local CSV to S3 first:
        aws s3 cp my_data.csv s3://my-bucket/data/my_data.csv
    Then pass --dataset s3://my-bucket/data/my_data.csv
"""

import argparse
import logging
import os
import time

logger = logging.getLogger(__name__)


def submit(
    ray_address: str,
    config_path: str,
    dataset_path: str,
    output_dir: str,
    pip_packages: list[str] | None = None,
    num_gpus: float | None = None,
    follow_logs: bool = True,
):
    """Submit a Ludwig training job to a Ray cluster.

    Args:
        ray_address: Ray Dashboard address, e.g. "http://ray-head:8265"
        config_path: Path to Ludwig YAML config (local file, will be uploaded)
        dataset_path: Dataset path accessible from the cluster (S3/GCS/NFS)
        output_dir: Where to save the trained model (S3/GCS/NFS recommended)
        pip_packages: Additional pip packages to install on the cluster
        num_gpus: Number of GPUs to request for the head node entrypoint
        follow_logs: Whether to stream logs until job completes
    """
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(ray_address)

    # Build runtime environment
    # The working_dir uploads local files (config + training script) to the cluster
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_env = {
        "working_dir": script_dir,
        # NOTE: If Ludwig is already installed on the cluster (recommended for
        # production), you can remove the pip section. Installing at job start
        # adds ~2-5 min of cold start time.
        "env_vars": {
            "CONFIG_PATH": os.path.basename(config_path),
            "DATASET_PATH": dataset_path,
            "OUTPUT_DIR": output_dir,
        },
    }

    if pip_packages:
        runtime_env["pip"] = pip_packages

    # Copy config to the script directory so it's included in working_dir upload
    import shutil

    config_dest = os.path.join(script_dir, os.path.basename(config_path))
    if os.path.abspath(config_path) != os.path.abspath(config_dest):
        shutil.copy2(config_path, config_dest)

    # Submit the job
    # NOTE: entrypoint_num_gpus reserves GPUs for the driver script. The actual
    # training workers request their own GPUs via Ludwig's Ray backend config.
    entrypoint = "python train_on_cluster.py"
    submit_kwargs = {"entrypoint": entrypoint, "runtime_env": runtime_env}
    if num_gpus is not None:
        submit_kwargs["entrypoint_num_gpus"] = num_gpus

    logger.info(f"Submitting job to {ray_address}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Output: {output_dir}")

    job_id = client.submit_job(**submit_kwargs)
    logger.info(f"Job submitted: {job_id}")

    if not follow_logs:
        print(f"Job ID: {job_id}")
        print(f"Check status: ray job status {job_id} --address {ray_address}")
        print(f"Stream logs:  ray job logs {job_id} --address {ray_address} --follow")
        return job_id

    # Stream logs and wait for completion
    # NOTE: This blocks until the job finishes. For long training runs, you may
    # prefer to use --no-follow and check status manually.
    print(f"Job {job_id} submitted. Streaming logs...\n")

    prev_logs = ""
    while True:
        status = client.get_job_status(job_id)
        logs = client.get_job_logs(job_id)

        # Print new log lines
        if logs != prev_logs:
            new_lines = logs[len(prev_logs) :]
            print(new_lines, end="", flush=True)
            prev_logs = logs

        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            break

        time.sleep(2)

    print(f"\nJob {job_id} finished with status: {status}")

    if status == JobStatus.FAILED:
        # Print any error info
        details = client.get_job_info(job_id)
        if hasattr(details, "error_type"):
            print(f"Error: {details.error_type}: {details.message}")
        return None

    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit a Ludwig training job to a Ray cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit to a local Ray cluster
  python submit_job.py --ray-address http://localhost:8265 \\
      --config config.yaml \\
      --dataset s3://my-bucket/train.csv \\
      --output-dir s3://my-bucket/results/

  # Submit to a KubeRay cluster
  python submit_job.py --ray-address http://ray-head.ray.svc:8265 \\
      --config config.yaml \\
      --dataset s3://my-bucket/train.csv \\
      --output-dir s3://my-bucket/results/

  # Install Ludwig on the cluster at job start (slower but no pre-install needed)
  python submit_job.py --ray-address http://ray-head:8265 \\
      --config config.yaml \\
      --dataset s3://my-bucket/train.csv \\
      --output-dir /shared/results/ \\
      --pip ludwig[distributed]
        """,
    )
    parser.add_argument(
        "--ray-address",
        required=True,
        help="Ray Dashboard address (e.g. http://ray-head:8265)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Ludwig YAML config (local file, will be uploaded to cluster)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path accessible from the cluster (s3://, gs://, /nfs/...)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/ludwig_results",
        help="Where to save results on the cluster (default: /tmp/ludwig_results)",
    )
    parser.add_argument(
        "--pip",
        nargs="*",
        default=None,
        dest="pip_packages",
        help="Additional pip packages to install on the cluster at job start",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=None,
        help="GPUs to reserve for the driver script",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Don't stream logs. Print job ID and exit.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    submit(
        ray_address=args.ray_address,
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        pip_packages=args.pip_packages,
        num_gpus=args.num_gpus,
        follow_logs=not args.no_follow,
    )


if __name__ == "__main__":
    main()
