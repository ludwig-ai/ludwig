import os
import uuid

import ray.train
import tqdm


class LudwigProgressBar:
    def __init__(self, report_to_ray, config, is_coordinator):
        self.id = str(uuid.uuid4())[-8:]
        self.config = config
        self.report_to_ray = report_to_ray
        self.total_steps = 0
        self.progress_bar = None
        self.is_coordinator = is_coordinator
        if not self.report_to_ray:
            if self.is_coordinator:
                self.progress_bar = tqdm.tqdm(**config)
        else:
            if "file" in self.config:
                self.config.pop("file")
            # TODO magdy: add a comment explaining why all processes 
            # need to report
            ray.train.report(
                progress_bar={
                    "id": self.id,
                    "config": self.config,
                    "update_by": 0,
                    "pid": os.getpid(),
                }
            )

    def update(self, steps):
        self.total_steps += steps
        if self.progress_bar:
            self.progress_bar.update(steps)
        else:
            ray.train.report(
                progress_bar={
                    "id": self.id,
                    "update_by": steps,
                    "pid": os.getpid(),
                }
            )

    def close(self):
        if self.progress_bar:
            self.progress_bar.close()
