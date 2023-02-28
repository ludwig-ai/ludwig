# Implements https://github.com/ray-project/ray/pull/30598 ahead of Ray 2.2 release.

import math
from typing import Any, Callable, Dict, Optional, Type, TYPE_CHECKING, Union

import ray
from ray.air.config import RunConfig
from ray.tune.execution.trial_runner import _ResumeConfig
from ray.tune.impl.tuner_internal import TunerInternal
from ray.tune.trainable import Trainable
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import _SELF, _TUNER_INTERNAL, Tuner
from ray.tune.utils.node import _force_on_current_node

if TYPE_CHECKING:
    from ray.train.trainer import BaseTrainer


class TunerRay210(Tuner):
    """HACK(geoffrey): This is a temporary fix to support Ray 2.1.0.

    Specifically, this Tuner ensures that TunerInternalRay210 is called by the class.
    For more details, see TunerInternalRay210.
    """

    def __init__(
        self,
        trainable: Optional[
            Union[
                str,
                Callable,
                Type[Trainable],
                "BaseTrainer",
            ]
        ] = None,
        *,
        param_space: Optional[Dict[str, Any]] = None,
        tune_config: Optional[TuneConfig] = None,
        run_config: Optional[RunConfig] = None,
        # This is internal only arg.
        # Only for dogfooding purposes. We can slowly promote these args
        # to RunConfig or TuneConfig as needed.
        # TODO(xwjiang): Remove this later.
        _tuner_kwargs: Optional[Dict] = None,
        _tuner_internal: Optional[TunerInternal] = None,
    ):
        """Configure and construct a tune run."""
        kwargs = locals().copy()
        self._is_ray_client = ray.util.client.ray.is_connected()
        if _tuner_internal:
            if not self._is_ray_client:
                self._local_tuner = kwargs[_TUNER_INTERNAL]
            else:
                self._remote_tuner = kwargs[_TUNER_INTERNAL]
        else:
            kwargs.pop(_TUNER_INTERNAL, None)
            kwargs.pop(_SELF, None)
            if not self._is_ray_client:
                self._local_tuner = TunerInternalRay210(**kwargs)
            else:
                self._remote_tuner = _force_on_current_node(ray.remote(num_cpus=0)(TunerInternalRay210)).remote(
                    **kwargs
                )

    @classmethod
    def restore(
        cls,
        path: str,
        resume_unfinished: bool = True,
        resume_errored: bool = False,
        restart_errored: bool = False,
    ) -> "Tuner":
        """Restores Tuner after a previously failed run.

        All trials from the existing run will be added to the result table. The
        argument flags control how existing but unfinished or errored trials are
        resumed.

        Finished trials are always added to the overview table. They will not be
        resumed.

        Unfinished trials can be controlled with the ``resume_unfinished`` flag.
        If ``True`` (default), they will be continued. If ``False``, they will
        be added as terminated trials (even if they were only created and never
        trained).

        Errored trials can be controlled with the ``resume_errored`` and
        ``restart_errored`` flags. The former will resume errored trials from
        their latest checkpoints. The latter will restart errored trials from
        scratch and prevent loading their last checkpoints.

        Args:
            path: The path where the previous failed run is checkpointed.
                This information could be easily located near the end of the
                console output of previous run.
                Note: depending on whether ray client mode is used or not,
                this path may or may not exist on your local machine.
            resume_unfinished: If True, will continue to run unfinished trials.
            resume_errored: If True, will re-schedule errored trials and try to
                restore from their latest checkpoints.
            restart_errored: If True, will re-schedule errored trials but force
                restarting them from scratch (no checkpoint will be loaded).
        """
        resume_config = _ResumeConfig(
            resume_unfinished=resume_unfinished,
            resume_errored=resume_errored,
            restart_errored=restart_errored,
        )

        if not ray.util.client.ray.is_connected():
            tuner_internal = TunerInternalRay210(restore_path=path, resume_config=resume_config)
            return TunerRay210(_tuner_internal=tuner_internal)
        else:
            tuner_internal = _force_on_current_node(ray.remote(num_cpus=0)(TunerInternalRay210)).remote(
                restore_path=path, resume_config=resume_config
            )
            return TunerRay210(_tuner_internal=tuner_internal)


class TunerInternalRay210(TunerInternal):
    """HACK(geoffrey): This is a temporary fix to support Ray 2.1.0.

    This TunerInternal ensures that a division by zero is avoided when running zero-CPU hyperopt trials.
    This is fixed in ray>=2.2 (but not ray<=2.1) here: https://github.com/ray-project/ray/pull/30598
    """

    def _expected_utilization(self, cpus_per_trial, cpus_total):
        num_samples = self._tune_config.num_samples
        if num_samples < 0:  # TODO: simplify this in Tune
            num_samples = math.inf
        concurrent_trials = self._tune_config.max_concurrent_trials or 0
        if concurrent_trials < 1:  # TODO: simplify this in Tune
            concurrent_trials = math.inf

        actual_concurrency = min(
            (
                (cpus_total // cpus_per_trial) if cpus_per_trial else 0,
                num_samples,
                concurrent_trials,
            )
        )
        return (actual_concurrency * cpus_per_trial) / (cpus_total + 0.001)
