"""some parts are inspired from https://github.com/Breakend/experiment-impact-
tracker/blob/master/experiment_impact_tracker/compute_tracker.py."""

import multiprocessing
import os
import shutil
import sys
import time
import traceback
from queue import Empty as EmptyQueueException
from statistics import mean
from typing import Any, Dict, Optional

import psutil
import torch
from gpustat.core import GPUStatCollection

from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_json, save_json

# disabling print because the following imports are verbose
f = open(os.devnull, "w")
sys.stdout = f
from experiment_impact_tracker.cpu.common import get_my_cpu_info
from experiment_impact_tracker.gpu.nvidia import get_gpu_info
from experiment_impact_tracker.py_environment.common import get_python_packages_and_versions

f.close()
sys.stdout = sys.__stdout__

STOP_MESSAGE = "stop"


def monitor(queue: multiprocessing.Queue, info: Dict[str, Any], output_dir: str, logging_interval: int) -> None:
    """Monitors hardware resource use as part of a separate process.

    Populate `info` with system specific metrics (GPU, CPU, RAM) at a `logging_interval` interval and saves the output
    in `output_dir`.

    Args:
        queue: queue from which we can push and retrieve messages sent to the child process.
        info: dictionary containing system resource usage information about the parent process.
        output_dir: directory where the contents of `info` will be saved.
        logging_interval: time interval at which we will poll the system for usage metrics.
    """
    for key in info["system"]:
        if "gpu_" in key:
            info["system"][key]["memory_used"] = []
    info["system"]["cpu_utilization"] = []
    info["system"]["ram_utilization"] = []

    while True:
        try:
            message = queue.get(block=False)
            if isinstance(message, str):
                if message == STOP_MESSAGE:
                    save_json(os.path.join(output_dir, info["tag"] + "_temp.json"), info)
                    return
            else:
                queue.put(message)
        except EmptyQueueException:
            pass
        if torch.cuda.is_available():
            gpu_infos = GPUStatCollection.new_query()
            for i, gpu_info in enumerate(gpu_infos):
                gpu_key = f"gpu_{i}"
                info["system"][gpu_key]["memory_used"].append(gpu_info.memory_used)
        info["system"]["cpu_utilization"].append(psutil.cpu_percent())
        info["system"]["ram_utilization"].append(psutil.virtual_memory().percent)
        time.sleep(logging_interval)


class ResourceUsageTracker:
    """Track system resource (hardware and software) usage.

    Attributes:
        tag: a string tag about the process that we're tracking. Examples: train, evaluate, preprocess, etc.
        output_dir: path where metrics are saved.
        logging_interval: time interval in seconds at which system is polled for resource usage.
        num_examples: number of examples of training or evaluation process.
    """

    def __init__(
        self,
        tag: str,
        output_dir: str,
        logging_interval: float = 1.0,
        num_examples: Optional[int] = None,
    ) -> None:
        if tag not in ["train", "evaluate", "preprocess"]:
            raise ValueError(f"{self.__class__.__name__} tag unrecognized. Please choose one from [train, evaluate, "
                             f"preprocess]")
        
        self.output_dir = output_dir
        self.tag = tag
        self.info = {"tag": self.tag, "system": {}}
        self.num_examples = num_examples
        self.logging_interval = logging_interval
        self.launched = False
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)

    def populate_static_information(self) -> None:
        """Populates the report with static software and hardware information."""
        self.info["ludwig_version"] = LUDWIG_VERSION
        self.info["start_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used

        # CPU information
        self.info["system"]["python_packages_and_versions"] = [
            str(package) for package in get_python_packages_and_versions()
        ]
        cpu_info = get_my_cpu_info()
        self.info["system"]["cpu_architecture"] = cpu_info["arch"]
        self.info["system"]["num_cpu"] = cpu_info["count"]
        self.info["system"]["cpu_name"] = cpu_info["brand_raw"]

        # GPU information
        if torch.cuda.is_available():
            gpu_infos = get_gpu_info()
            for i, gpu_info in enumerate(gpu_infos):
                gpu_key = f"gpu_{i}"
                self.info["system"][gpu_key] = {}
                self.info["system"][gpu_key]["name"] = gpu_info["name"]
                self.info["system"][gpu_key]["total_memory"] = gpu_info["total_memory"]
                self.info["system"][gpu_key]["driver_version"] = gpu_info["driver_version"]
                self.info["system"][gpu_key]["cuda_version"] = gpu_info["cuda_version"]

        self.info["start_time"] = time.time()
        self.info["num_examples"] = self.num_examples

    def __enter__(self):
        """Populates static information and forks process to monitor resource usage."""
        if self.launched:
            raise ValueError("Tracker already launched.")

        self.populate_static_information()
        try:
            ctx = multiprocessing.get_context("fork")
            self.queue = ctx.Queue()
            self.p = ctx.Process(
                target=monitor,
                args=(
                    self.queue,
                    self.info,
                    self.output_dir,
                    self.logging_interval,
                ),
            )
            self.p.start()
            self.launched = True
        except Exception as _:
            self.launched = False
            ex_type, ex_value, tb = sys.exc_info()
            print("Encountered exception when launching tracker.")
            print("".join(traceback.format_tb(tb)))
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Waits for monitoring process to exit.

        Computes and postprocesses more metrics. Saves report.
        """
        self.queue.put(STOP_MESSAGE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.p.join()

        self.info = load_json(os.path.join(self.output_dir, self.info["tag"] + "_temp.json"))
        os.remove(os.path.join(self.output_dir, self.info["tag"] + "_temp.json"))

        self.info["end_time"] = time.time()
        self.info[f"{self.tag}_total_duration"] = self.info["end_time"] - self.info["start_time"]

        if self.num_examples:
            self.info["examples_per_second"] = self.num_examples / self.info[f"{self.tag}_total_duration"]
        self.info["end_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used
        self.info["disk_footprint"] = self.info["end_disk_usage"] - self.info["start_disk_usage"]

        for key in self.info["system"]:
            if "gpu_" in key:
                self.info["system"][key]["max_memory_used"] = max(self.info["system"][key]["memory_used"])
        self.info["system"]["max_cpu_utilization"] = max(self.info["system"]["cpu_utilization"])
        self.info["system"]["max_ram_utilization"] = max(self.info["system"]["ram_utilization"])

        self.info["system"]["average_cpu_utilization"] = mean(self.info["system"]["cpu_utilization"])
        self.info["system"]["average_ram_utilization"] = mean(self.info["system"]["ram_utilization"])

        save_json(os.path.join(self.output_dir, self.info["tag"] + "_resource_usage_metrics.json"), self.info)
