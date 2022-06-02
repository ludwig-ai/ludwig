import uuid
import os

import tqdm

try:
    import ray.train as rt  # noqa: E402
except ImportError:
    rt = None



class LudwigProgressBarActions:
    CREATE = 'create'
    UPDATE = 'update'
    CLOSE = 'close'


class LudwigProgressBar:
    def __init__(self, report_to_ray, config, is_coordinator):
        if report_to_ray and rt is None:
            raise ValueError("Set report_to_ray=True but ray is not installed. Run `pip install ray`")

        self.id = str(uuid.uuid4())[-8:]

        self.report_to_ray = report_to_ray
        self.is_coordinator = is_coordinator
        self.config = config

        self.total_steps = 0
        self.progress_bar = None

        if not self.report_to_ray:
            if self.is_coordinator:
                self.progress_bar = tqdm.tqdm(**config)
        else:
            if "file" in self.config:
                self.config.pop("file")
            # All processes need to call ray.train.report since ray has a lock that blocks 
            # a process when calling report if there are processes that haven't called it. Similar
            # to a distributed checkpoint. Therefore we pass the flag to the driver
            rt.report(
                progress_bar={
                    "id": self.id,
                    "config": self.config,
                    "action": LudwigProgressBarActions.CREATE,
                    "is_coordinator": self.is_coordinator,
                }
            )

    def update(self, steps):
        self.total_steps += steps
        if self.progress_bar:
            self.progress_bar.update(steps)
        elif self.report_to_ray:
            rt.report(
                progress_bar={
                    "id": self.id,
                    "update_by": steps,
                    "is_coordinator": self.is_coordinator,
                    "action": LudwigProgressBarActions.UPDATE,
                }
            )

    def close(self):
        if self.progress_bar:
            self.progress_bar.close()
        elif self.report_to_ray:
            rt.report(
                progress_bar={
                    "id": self.id,
                    "is_coordinator": self.is_coordinator,
                    "action": LudwigProgressBarActions.CLOSE,
                }
            )

