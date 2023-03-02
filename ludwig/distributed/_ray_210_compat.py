from ray.air.result import Result
from ray.train.base_trainer import TrainingFailedError
from ray.train.horovod import HorovodTrainer

from ludwig.backend._ray210_compat import TunerRay210


class HorovodTrainerRay210(HorovodTrainer):
    """HACK(geoffrey): This is a temporary fix to support Ray 2.1.0.

    Specifically, this Trainer ensures that TunerRay210 is called by the class.
    For more details, see TunerRay210.
    """

    def fit(self) -> Result:
        """Runs training.

        Returns:
            A Result object containing the training result.

        Raises:
            TrainingFailedError: If any failures during the execution of
            ``self.as_trainable()``.
        """
        from ray.tune.error import TuneError

        trainable = self.as_trainable()

        tuner = TunerRay210(trainable=trainable, run_config=self.run_config)
        result_grid = tuner.fit()
        assert len(result_grid) == 1
        try:
            result = result_grid[0]
            if result.error:
                raise result.error
        except TuneError as e:
            raise TrainingFailedError from e
        return result
