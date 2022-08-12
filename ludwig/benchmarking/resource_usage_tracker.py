"""some parts are inspired from https://github.com/Breakend/experiment-impact-
tracker/blob/master/experiment_impact_tracker/compute_tracker.py."""

import contextlib
import os
import shutil
import sys
import threading
import time
import traceback
from queue import Empty as EmptyQueueException
from queue import Queue
from statistics import mean
from typing import Any, Dict, Optional

import psutil
import torch
from gpustat.core import GPUStatCollection

from ludwig.benchmarking.reporting import export_metrics_from_torch_profiler
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_json, save_json

# disabling print because the following imports are verbose
f = open(os.devnull, "w")
sys.stdout = f
from experiment_impact_tracker.cpu.common import get_my_cpu_info  # noqa E402
from experiment_impact_tracker.gpu.nvidia import get_gpu_info  # noqa E402
from experiment_impact_tracker.py_environment.common import get_python_packages_and_versions  # noqa E402

f.close()
sys.stdout = sys.__stdout__

STOP_MESSAGE = "stop"


def monitor(
    queue: Queue, info: Dict[str, Any], output_dir: str, logging_interval: int, cuda_is_available: bool
) -> None:
    """Monitors hardware resource use.

    Populate `info` with system specific metrics (CPU/CUDA, CPU/CUDA memory) at a `logging_interval` interval and saves the output
    in `output_dir`.

    Args:
        queue: queue from which we can push and retrieve messages sent to the function targeted by the thread.
        info: dictionary containing system resource usage information about the running process.
        output_dir: directory where the contents of `info` will be saved.
        logging_interval: time interval at which we will poll the system for usage metrics.
        cuda_is_available: stores torch.cuda.is_available().
    """
    for key in info:
        if "cuda_" in key:
            info[key]["memory_used"] = []
    info["cpu_utilization"] = []
    info["cpu_memory_usage"] = []

    # get the pid of the parent process.
    tracked_process = psutil.Process(os.getpid())

    # will return a meaningless 0 value on the first call because `interval` arg is set to None.
    tracked_process.cpu_percent(interval=logging_interval)

    while True:
        time.sleep(logging_interval)
        try:
            message = queue.get(block=False)
            if isinstance(message, str):
                if message == STOP_MESSAGE:
                    # synchronize CUDA to get accurate timing for jobs running on GPU.
                    if cuda_is_available:
                        torch.cuda.synchronize()
                    queue.put(info)
                    save_json(os.path.join(output_dir, info["tag"] + "_temp.json"), info)
                    return
            else:
                queue.put(message)
        except EmptyQueueException:
            pass
        if cuda_is_available:
            gpu_infos = GPUStatCollection.new_query()
            for i, gpu_info in enumerate(gpu_infos):
                gpu_key = f"cuda_{i}"
                info[gpu_key]["memory_used"].append(gpu_info.memory_used)
        with tracked_process.oneshot():
            info["cpu_utilization"].append(tracked_process.cpu_percent())
            info["cpu_memory_usage"].append(tracked_process.memory_full_info().uss)


class ResourceUsageTracker(contextlib.ContextDecorator):
    """Track system resource (hardware and software) usage.

    Attributes:
        tag: a string tag about the process that we're tracking. Examples: train, evaluate, preprocess, etc.
        output_dir: path where metrics are saved.
        logging_interval: time interval in seconds at which system is polled for resource usage.
    """

    def __init__(
        self,
        tag: str,
        use_torch_profiler: bool,
        output_dir: str,
        logging_interval: float = 0.1,
    ) -> None:
        self.output_dir = output_dir
        self.tag = tag
        self.info = {"tag": self.tag}
        self.logging_interval = logging_interval
        self.launched = False
        self.cuda_is_available = torch.cuda.is_available()
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        self.use_torch_profiler = use_torch_profiler

        if self.use_torch_profiler:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda_is_available:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.torch_profiler = torch.profiler.profile(activities=activities, profile_memory=True)
            self.torch_record_function = torch.profiler.record_function(self.tag)

    def populate_static_information(self) -> None:
        """Populates the report with static software and hardware information."""
        self.info["ludwig_version"] = LUDWIG_VERSION
        self.info["start_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used

        # CPU information
        # self.info["python_packages_and_versions"] = [
        #     str(package) for package in get_python_packages_and_versions()
        # ]
        cpu_info = get_my_cpu_info()
        self.info["cpu_architecture"] = cpu_info["arch"]
        self.info["num_cpu"] = cpu_info["count"]
        self.info["cpu_name"] = cpu_info["brand_raw"]
        self.info["cpu_memory_available"] = psutil.virtual_memory().available

        # GPU information
        if self.cuda_is_available:
            gpu_infos = get_gpu_info()
            for i, gpu_info in enumerate(gpu_infos):
                gpu_key = f"cuda_{i}"
                self.info[gpu_key] = {}
                self.info[gpu_key]["name"] = gpu_info["name"]
                self.info[gpu_key]["total_memory"] = gpu_info["total_memory"]
                self.info[gpu_key]["driver_version"] = gpu_info["driver_version"]
                self.info[gpu_key]["cuda_version"] = gpu_info["cuda_version"]

        # recording in microseconds to be in line with torch profiler time recording.
        self.info["start_time"] = time.perf_counter_ns() / 1000

    def __enter__(self):
        """Populates static information and monitors resource usage."""
        if self.launched:
            raise ValueError("Tracker already launched.")

        self.populate_static_information()
        if self.use_torch_profiler:
            # contextlib.ExitStack gracefully handles situations where __enter__ or __exit__ calls throw exceptions.
            with contextlib.ExitStack() as ctx_exit_stack:
                ctx_exit_stack.enter_context(self.torch_profiler)
                ctx_exit_stack.enter_context(self.torch_record_function)
                self._ctx_exit_stack = ctx_exit_stack.pop_all()
        try:
            self.queue = Queue()
            self.t = threading.Thread(
                target=monitor,
                args=(
                    self.queue,
                    self.info,
                    self.output_dir,
                    self.logging_interval,
                    self.cuda_is_available,
                ),
            )
            self.t.start()
            self.launched = True
        except Exception:
            self.launched = False
            ex_type, ex_value, tb = sys.exc_info()
            print("Encountered exception when launching tracker thread.")
            print("".join(traceback.format_tb(tb)))
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Joins monitoring thread.

        Computes and postprocesses more metrics. Saves report.
        """
        try:
            self.queue.put(STOP_MESSAGE)
            self.t.join()
            self.info = self.queue.get()
            # recording in microseconds to be in line with torch profiler time recording.
            self.info["end_time"] = time.perf_counter_ns() / 1000
            self.launched = False
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            print("Encountered exception when joining tracker thread.")
            print("".join(traceback.format_tb(tb)))
        finally:
            if self.use_torch_profiler:
                self._ctx_exit_stack.close()

        self.info["total_duration"] = self.info.pop("end_time") - self.info.pop("start_time")
        self.info["end_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used
        self.info["disk_footprint"] = self.info.pop("end_disk_usage") - self.info.pop("start_disk_usage")

        for key in self.info:
            if "cuda_" in key:
                self.info[key]["max_memory_used"] = max(self.info[key]["memory_used"])
        self.info["max_cpu_utilization"] = max(self.info["cpu_utilization"], default=None)
        self.info["max_cpu_memory_utilization"] = max(self.info["cpu_memory_usage"], default=None)

        if self.info["cpu_utilization"]:
            self.info["average_cpu_utilization"] = mean(self.info.pop("cpu_utilization"))
        if self.info["cpu_memory_usage"]:
            self.info["average_cpu_memory_utilization"] = mean(self.info.pop("cpu_memory_usage"))

        # todo (Wael) clean up
        torch_usage_metrics = export_metrics_from_torch_profiler([self.tag], self.torch_profiler, self.output_dir)[
            self.tag
        ]["runs"][0]
        for key, value in torch_usage_metrics.items():
            self.info[key] = value

        save_json(os.path.join(self.output_dir, self.info["tag"] + "_resource_usage_metrics.json"), self.info)
