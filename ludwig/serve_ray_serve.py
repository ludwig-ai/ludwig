"""Ray Serve deployment wrapper for a Ludwig model (Phase 6.8).

Exposes a :class:`LudwigDeployment` class that wraps a trained ``LudwigModel`` behind a
Ray Serve HTTP endpoint with the same ``/predict`` and ``/batch_predict`` payload shape
as the FastAPI server (``ludwig.serve_v2``).  This lets operators deploy Ludwig models
across a Ray cluster with autoscaling, traffic splitting, and rolling rollout, rather
than managing single-process FastAPI instances by hand.

Usage (inside a Ray-aware process)::

    from ludwig.serve_ray_serve import deploy_ludwig_model

    handle = deploy_ludwig_model(
        model_path="/path/to/model",
        name="sentiment",
        num_replicas=2,
        ray_actor_options={"num_gpus": 1},
    )

The deployment can then be called programmatically via ``handle.remote(payload)`` or
through the HTTP endpoint that Ray Serve spins up automatically.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_ray_serve():
    """Import ``ray.serve`` lazily with a clear error if unavailable."""
    try:
        from ray import serve
    except ImportError as exc:
        raise ImportError(
            "Ray Serve is required for ludwig.serve_ray_serve. Install with: " "pip install 'ludwig[distributed]'"
        ) from exc
    return serve


def make_ludwig_deployment_class(num_replicas: int = 1, ray_actor_options: dict | None = None):
    """Build a Ray Serve deployment class wrapping ``LudwigModel.load`` + ``predict``.

    The class is constructed at call time so that importing this module on a machine
    without ray.serve installed does not fail — only :func:`deploy_ludwig_model` and
    this helper actually touch the Ray Serve API.

    Args:
        num_replicas: number of actor replicas to run.
        ray_actor_options: ``@serve.deployment(ray_actor_options=...)`` overrides for
            per-replica resources (e.g. ``{"num_gpus": 1}``).

    Returns:
        A ``@serve.deployment``-decorated class ready to ``.bind(model_path)``.
    """
    serve = _import_ray_serve()

    @serve.deployment(num_replicas=num_replicas, ray_actor_options=ray_actor_options or {})
    class LudwigDeployment:
        def __init__(self, model_path: str) -> None:
            from ludwig.api import LudwigModel

            self._model = LudwigModel.load(model_path)
            logger.info("Ray Serve replica loaded Ludwig model from %s", model_path)

        async def __call__(self, request: Any) -> dict | list:
            # Accept either a single dict record or a list of dict records.
            payload = await request.json() if hasattr(request, "json") else request
            if isinstance(payload, list):
                import pandas as pd

                preds, _ = self._model.predict(dataset=pd.DataFrame(payload))
                return {"predictions": preds.to_dict(orient="records")}
            else:
                import pandas as pd

                preds, _ = self._model.predict(dataset=pd.DataFrame([payload]))
                records = preds.to_dict(orient="records")
                if not records:
                    return {}
                return records[0]

        async def predict(self, payload: dict | list[dict]) -> dict | list[dict]:
            """Programmatic entry point for Ray Serve ``handle.predict.remote(...)`` calls."""
            import pandas as pd

            records = payload if isinstance(payload, list) else [payload]
            preds, _ = self._model.predict(dataset=pd.DataFrame(records))
            return preds.to_dict(orient="records")

    return LudwigDeployment


def deploy_ludwig_model(
    model_path: str,
    *,
    name: str = "ludwig",
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    route_prefix: str | None = None,
):
    """Deploy a trained Ludwig model as a Ray Serve application.

    Args:
        model_path: path to the trained Ludwig model directory.
        name: Ray Serve application name; also the HTTP route suffix (``/{name}``).
        num_replicas: number of replicas for the deployment.
        ray_actor_options: per-replica resources (e.g. ``{"num_gpus": 1}``).
        route_prefix: explicit URL prefix; defaults to ``/{name}``.

    Returns:
        The deployed application's handle (a ``DeploymentHandle``). Call
        ``handle.predict.remote(payload)`` to issue predictions programmatically.
    """
    serve = _import_ray_serve()
    deployment_cls = make_ludwig_deployment_class(num_replicas=num_replicas, ray_actor_options=ray_actor_options)

    app = deployment_cls.bind(model_path)
    handle = serve.run(app, name=name, route_prefix=route_prefix or f"/{name}")
    logger.info(
        "Deployed Ludwig model %r as Ray Serve app %r (%d replica(s))",
        model_path,
        name,
        num_replicas,
    )
    return handle
