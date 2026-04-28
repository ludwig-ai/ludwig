"""Deploy a trained Ludwig model with Ray Serve.

Usage:
    # Single replica, CPU
    python deploy.py --model_path ./results/my_model

    # Two GPU replicas
    python deploy.py \
        --model_path ./results/my_model \
        --num_replicas 2 \
        --gpu

    # Custom name / port
    python deploy.py \
        --model_path ./results/my_model \
        --name sentiment \
        --port 8080

After deploying, send predictions:

    # Single record
    curl -s -X POST http://localhost:8000/ludwig \\
        -H "Content-Type: application/json" \\
        -d '{"text": "I love this product!", "stars": 5}'

    # Batch (list of records)
    curl -s -X POST http://localhost:8000/ludwig \\
        -H "Content-Type: application/json" \\
        -d '[{"text": "great"}, {"text": "terrible"}]'
"""

import argparse
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="Path to trained Ludwig model directory")
    p.add_argument("--name", default="ludwig", help="Ray Serve application name (also URL prefix)")
    p.add_argument("--num_replicas", type=int, default=1, help="Number of Ray Serve replicas")
    p.add_argument("--gpu", action="store_true", help="Request 1 GPU per replica")
    p.add_argument("--port", type=int, default=8000, help="Ray dashboard / Serve port")
    p.add_argument("--block", action="store_true", help="Keep running until interrupted")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import ray
    from ray import serve

    from ludwig.serve_ray_serve import deploy_ludwig_model

    ray_actor_options = {"num_gpus": 1} if args.gpu else {}

    print("Initialising Ray …")
    ray.init(ignore_reinit_error=True)

    print(f"Deploying Ludwig model from {args.model_path!r} …")
    deploy_ludwig_model(
        model_path=args.model_path,
        name=args.name,
        num_replicas=args.num_replicas,
        ray_actor_options=ray_actor_options,
    )

    url = f"http://localhost:{args.port}/{args.name}"
    print(f"\nDeployment live at: {url}")
    print(f"  POST {url}  with a JSON record or list of records to get predictions.")

    if args.block:
        print("Press Ctrl-C to stop …")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Ray Serve …")
            serve.shutdown()
            ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
