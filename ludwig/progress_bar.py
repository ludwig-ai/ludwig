import tqdm

try:
    import ray.train as rt
except ImportError:
    rt = None


class LudwigProgressBar:
    """Progress bar that works both locally and inside Ray Train workers.

    When ``report_to_ray=True`` the bar is silently suppressed so that Ray
    worker subprocesses do not spam the driver log with tqdm escape codes, and
    — critically — so that ``rt.report()`` is *not* called on every training
    step.  Calling ``rt.report()`` every batch costs ~1.9 s per call (it
    requires a round-trip through the Ray GCS) and completely dominates
    wall-clock training time at ~2 s/batch overhead vs ~0.3 s of actual GPU
    compute.  Training metrics are already reported at eval/checkpoint time via
    the proper ``rt.report(checkpoint=...)`` call in the backend; per-batch
    progress updates via Ray Train are unnecessary.
    """

    def __init__(
        self,
        report_to_ray: bool,
        config: dict,
        is_coordinator: bool,
    ) -> None:
        self.report_to_ray = report_to_ray
        self.is_coordinator = is_coordinator
        self.config = config
        self.total_steps = 0
        self.progress_bar = None

        if not report_to_ray and is_coordinator:
            self.progress_bar = tqdm.tqdm(**config)

    def set_postfix(self, ordered_dict: dict = None, **kwargs) -> None:
        if self.progress_bar:
            self.progress_bar.set_postfix(ordered_dict, **kwargs)

    def update(self, steps: int) -> None:
        self.total_steps += steps
        if self.progress_bar:
            self.progress_bar.update(steps)

    def close(self) -> None:
        if self.progress_bar:
            self.progress_bar.close()
