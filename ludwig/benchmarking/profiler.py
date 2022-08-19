"""some parts are inspired from https://github.com/Breakend/experiment-impact-
tracker/blob/master/experiment_impact_tracker/compute_tracker.py."""

import contextlib
import glob
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from queue import Empty as EmptyQueueException
from queue import Queue
from statistics import mean
from typing import Any, Dict

import psutil
import torch
from gpustat.core import GPUStatCollection

from ludwig.benchmarking.reporting import get_metrics_from_torch_profiler
from ludwig.constants import LUDWIG_TAG
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import save_json

# disabling print because the following imports are verbose
f = open(os.devnull, "w")
sys.stdout = f
from experiment_impact_tracker.cpu.common import get_my_cpu_info  # noqa E402
from experiment_impact_tracker.gpu.nvidia import get_gpu_info  # noqa E402
from experiment_impact_tracker.py_environment.common import get_python_packages_and_versions  # noqa E402

f.close()
sys.stdout = sys.__stdout__

STOP_MESSAGE = "stop"


def monitor(queue: Queue, info: Dict[str, Any], logging_interval: int, cuda_is_available: bool) -> None:
    """Monitors hardware resource use.

    Populate `info` with system specific metrics (CPU/CUDA, CPU/CUDA memory) at a `logging_interval` interval
    and saves the output in `output_dir`.

    Args:
        queue: queue from which we can push and retrieve messages sent to the function targeted by the thread.
        info: dictionary containing system resource usage information about the running process.
        logging_interval: time interval at which we will poll the system for usage metrics.
        cuda_is_available: stores torch.cuda.is_available().
    """
    # get the pid of the parent process.
    tracked_process = psutil.Process(os.getpid())

    # will return a meaningless 0 value on the first call because `interval` arg is set to None.
    tracked_process.cpu_percent(interval=logging_interval)
    with tracked_process.oneshot():
        info["cpu_utilization"] = [tracked_process.cpu_percent()]
        info["cpu_memory_usage"] = [tracked_process.memory_full_info().uss]
        try:
            info["num_accessible_cpus"] = len(tracked_process.cpu_affinity())
        except Exception:
            pass

    while True:
        try:
            message = queue.get(block=False)
            if isinstance(message, str):
                if message == STOP_MESSAGE:
                    # synchronize CUDA to get accurate timing for jobs running on GPU.
                    if cuda_is_available:
                        torch.cuda.synchronize()
                    queue.put(info)
                    return
            else:
                queue.put(message)
        except EmptyQueueException:
            pass
        if cuda_is_available:
            gpu_infos = GPUStatCollection.new_query()
            for i, gpu_info in enumerate(gpu_infos):
                gpu_key = f"cuda_{i}"
                info[f"{gpu_key}_memory_used"].append(gpu_info.memory_used)
        with tracked_process.oneshot():
            info["cpu_utilization"].append(tracked_process.cpu_percent())
            info["cpu_memory_usage"].append(tracked_process.memory_full_info().uss)
        time.sleep(logging_interval)


class LudwigProfiler(contextlib.ContextDecorator):
    """Track system resource (hardware and software) usage.

    Warning: If `use_torch_profiler=True` while profiling on CUDA, it's not possible to benchmark DataLoaders
    with `num_workers > 0` due to CUDA multiprocessing limitations. See warning under `profile` class
    definition: https://github.com/pytorch/pytorch/blob/master/torch/autograd/profiler.py

    Attributes:
        tag: a string tag about the process that we're tracking. Examples: train, evaluate, preprocess, etc.
        output_dir: path where metrics are saved.
        logging_interval: time interval in seconds at which system is polled for resource usage.
    """

    def __init__(self, tag: str, use_torch_profiler: bool, output_dir: str, logging_interval: float = 0.1) -> None:
        self.tag = tag
        self._tag = LUDWIG_TAG + self.tag
        self.use_torch_profiler = use_torch_profiler
        self.output_dir = output_dir
        self.logging_interval = logging_interval
        self.cuda_is_available = torch.cuda.is_available()
        self.launched = False
        if self.use_torch_profiler:
            self.profiler_activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda_is_available:
                self.profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)

    def _init_tracker_info(self):
        """Initialize new self.info, self.torch_profiler, and self.torch_record_function instances.

        Important to call this in __enter__ if the user decides not to create a new class instance and therefore
        __init__ wouldn't be called.
        """
        self.info = {"code_block_tag": self.tag}
        if self.use_torch_profiler:
            self.torch_profiler = torch.profiler.profile(activities=self.profiler_activities, profile_memory=True)
            self.torch_record_function = torch.profiler.record_function(self._tag)

    def populate_static_information(self) -> None:
        """Populates the report with static software and hardware information."""
        self.info["ludwig_version"] = LUDWIG_VERSION
        self.info["start_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used

        # CPU information
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
                self.info[f"{gpu_key}_memory_used"] = []
                self.info[f"{gpu_key}_name"] = gpu_info["name"]
                self.info[f"{gpu_key}_total_memory"] = gpu_info["total_memory"]
                self.info[f"{gpu_key}_driver_version"] = gpu_info["driver_version"]
                self.info[f"{gpu_key}_cuda_version"] = gpu_info["cuda_version"]

        # recording in microseconds to be in line with torch profiler time recording.
        self.info["start_time"] = time.perf_counter_ns() / 1000

    def __enter__(self):
        """Populates static information and monitors resource usage."""
        if self.launched:
            raise RuntimeError("LudwigProfiler already launched. You can't use the same instance.")

        self._init_tracker_info()
        self.populate_static_information()

        if self.use_torch_profiler:
            # contextlib.ExitStack gracefully handles situations where __enter__ or __exit__ calls throw exceptions.
            with contextlib.ExitStack() as ctx_exit_stack:
                try:
                    # Launch torch.profiler to track PyTorch operators.
                    ctx_exit_stack.enter_context(self.torch_profiler)
                except RuntimeError:
                    # PyTorch profiler is already enabled on this thread.
                    # Using the running PyTorch profiler to track events.
                    self.torch_profiler = None

                ctx_exit_stack.enter_context(self.torch_record_function)
                self._ctx_exit_stack = ctx_exit_stack.pop_all()
        try:
            # Starting thread to monitor system resource usage.
            self.queue = Queue()
            self.t = threading.Thread(
                target=monitor,
                args=(
                    self.queue,
                    self.info,
                    self.logging_interval,
                    self.cuda_is_available,
                ),
            )
            self.t.start()
            self.launched = True
        except Exception:
            self.launched = False
            ex_type, ex_value, tb = sys.exc_info()
            logging.error("Encountered exception when launching tracker thread.")
            logging.error("".join(traceback.format_tb(tb)))
            raise RuntimeError

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop profiling, postprocess and export resource usage metrics."""
        try:
            self.queue.put(STOP_MESSAGE)
            self.t.join()
            self.info = self.queue.get()
            # recording in microseconds to be in line with torch profiler time recording.
            self.info["end_time"] = time.perf_counter_ns() / 1000
            self.launched = False
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            logging.error("Encountered exception when joining tracker thread.")
            logging.error("".join(traceback.format_tb(tb)))
        finally:
            if self.use_torch_profiler:
                self._ctx_exit_stack.close()
        self._export_system_metrics()
        self._export_torch_metrics()

    def _export_system_metrics(self):
        """Export system resource usage metrics (no torch operators)."""
        self.info["total_execution_time"] = self.info.pop("end_time") - self.info.pop("start_time")
        self.info["end_disk_usage"] = shutil.disk_usage(os.path.expanduser("~")).used
        self.info["disk_footprint"] = self.info.pop("end_disk_usage") - self.info.pop("start_disk_usage")

        all_keys = list(self.info.keys())
        for key in all_keys:
            if "cuda_" in key and "_memory_used" in key:
                cuda_max_memory_key = key.replace("_memory_used", "_max_memory_used")
                self.info[cuda_max_memory_key] = max(self.info[key], default=0)
                cuda_average_memory_key = key.replace("_memory_used", "_average_memory_used")
                if self.info[key]:
                    self.info[cuda_average_memory_key] = mean(self.info.pop(key))
                else:
                    self.info.pop(key)
                    self.info[cuda_average_memory_key] = 0

        self.info["max_cpu_utilization"] = max(self.info["cpu_utilization"], default=0)
        self.info["max_cpu_memory_usage"] = max(self.info["cpu_memory_usage"], default=0)

        if self.info["cpu_utilization"]:
            self.info["average_cpu_utilization"] = mean(self.info.pop("cpu_utilization"))
        else:
            self.info.pop("cpu_utilization")
            self.info["average_cpu_utilization"] = 0
        if self.info["cpu_memory_usage"]:
            self.info["average_cpu_memory_usage"] = mean(self.info.pop("cpu_memory_usage"))
        else:
            self.info.pop("cpu_memory_usage")
            self.info["average_cpu_memory_usage"] = 0

        temp_dir = os.path.join(self.output_dir, "system_resource_usage", self.info["code_block_tag"])
        os.makedirs(temp_dir, exist_ok=True)
        num_prev_runs = len(glob.glob(os.path.join(temp_dir, "run_*.json")))
        file_name = os.path.join(temp_dir, f"run_{num_prev_runs}.json")
        save_json(file_name, self.info)

    def _reformat_torch_usage_metrics_tags(self, torch_usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
        reformatted_dict = {}
        for key, value in torch_usage_metrics.items():
            assert key.startswith(LUDWIG_TAG)
            reformatted_key = key[len(LUDWIG_TAG) :]
            reformatted_dict[reformatted_key] = value
        return reformatted_dict

    def _export_torch_metrics(self):
        """Export resource usage metrics of torch operators."""
        if self.torch_profiler:
            torch_usage_metrics = get_metrics_from_torch_profiler(self.torch_profiler)
            torch_usage_metrics = self._reformat_torch_usage_metrics_tags(torch_usage_metrics)
            for tag, runs in torch_usage_metrics.items():
                temp_dir = os.path.join(self.output_dir, "torch_ops_resource_usage", tag)
                os.makedirs(temp_dir, exist_ok=True)
                for run in runs:
                    num_prev_runs = len(glob.glob(os.path.join(temp_dir, "run_*.json")))
                    save_json(os.path.join(temp_dir, f"run_{num_prev_runs}.json"), run)
