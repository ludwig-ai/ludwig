"""
some parts are inspired from
https://github.com/Breakend/experiment-impact-tracker/blob/master/experiment_impact_tracker/compute_tracker.py
"""

import psutil
import shutil
import os
import time
import torch
import multiprocessing
import traceback
import sys

from queue import Empty as EmptyQueueException
from gpustat.core import GPUStatCollection
from experiment_impact_tracker.py_environment.common import get_python_packages_and_versions
from experiment_impact_tracker.gpu.nvidia import get_gpu_info
from experiment_impact_tracker.cpu.common import get_my_cpu_info

from ludwig.utils.data_utils import save_json, load_json
from ludwig.utils.misc_utils import processify
from ludwig.globals import LUDWIG_VERSION

STOP_MESSAGE = 'stop'


@processify
def monitor(queue, info, output_dir, logging_interval):
    """
    Monitors hardware resource use as part of a separate process.

    Populate `info` with system specific metrics (GPU, CPU, RAM, swap) at a `logging_interval` interval
    and saves the output in `output_dir`.
    """
    for key in info['system']:
        if 'gpu_' in key:
            info['system'][key]['memory_used'] = []
    info['system']['cpu_utilization'] = []
    info['system']['ram_utilization'] = []
    info['system']['swap_utilization'] = []

    while True:
        try:
            message = queue.get(block=False)
            if isinstance(message, str):
                if message == STOP_MESSAGE:
                    save_json(os.path.join(output_dir, info['tag'] + "_temp.json"), info)
                    return
            else:
                queue.put(message)
        except EmptyQueueException:
            pass

        gpu_infos = GPUStatCollection.new_query()
        for i, gpu_info in enumerate(gpu_infos):
            gpu_key = 'gpu_{}'.format(i)
            info['system'][gpu_key]['memory_used'].append(gpu_info.memory_used)
        info['system']['cpu_utilization'].append(psutil.cpu_percent())
        info['system']['ram_utilization'].append(psutil.virtual_memory().percent)
        info['system']['swap_utilization'].append(psutil.swap_memory().percent)
        time.sleep(logging_interval)


class Tracker:
    """
    Track system resource (hardware and software) usage by a chunk of code.
    """

    def __init__(self, tag, output_dir, logging_interval=1, num_batches=None, num_examples=None):
        """
        tag: one of `train` or `evaluate`.
        output_dir: path where metrics are saved.
        logging_interval: time interval in seconds at which system is polled for resource usage.
        num_batches: number of batches of training or evaluation process.
        num_batches: number of examples of training or evaluation process.
        """
        self.output_dir = output_dir
        self.tag = tag
        self.info = {'tag': self.tag, 'system': {}}
        self.num_batches = num_batches
        self.num_examples = num_examples
        self.logging_interval = logging_interval
        self.launched = False
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)

    def populate_static_information(self):
        """
        Populates the report with static software and hardware information.
        """
        self.info['ludwig_version'] = LUDWIG_VERSION
        self.info['start_disk_usage'] = shutil.disk_usage(os.path.expanduser('~')).used

        # CPU information
        self.info['system']['python_packages_and_versions'] = [str(package) for package in
                                                               get_python_packages_and_versions()]
        cpu_info = get_my_cpu_info()
        self.info['system']['cpu_architecture'] = cpu_info['arch']
        self.info['system']['num_cpu'] = cpu_info['count']
        self.info['system']['cpu_name'] = cpu_info['brand_raw']

        # GPU information
        gpu_infos = get_gpu_info()
        for i, gpu_info in enumerate(gpu_infos):
            gpu_key = 'gpu_{}'.format(i)
            self.info['system'][gpu_key] = {}
            self.info['system'][gpu_key]['name'] = gpu_info['name']
            self.info['system'][gpu_key]['total_memory'] = gpu_info['total_memory']
            self.info['system'][gpu_key]['driver_version'] = gpu_info['driver_version']
            self.info['system'][gpu_key]['cuda_version'] = gpu_info['cuda_version']

        torch.cuda.synchronize()
        self.info['start_time'] = time.time()
        self.info['num_examples'] = self.num_examples

    def __enter__(self):
        """
        Populates static information and forks process to monitor resource
        usage.
        """
        if self.launched:
            raise ValueError('Tracker already launched.')

        self.populate_static_information()

        try:
            multiprocessing.set_start_method("fork")
        except RuntimeError:
            pass
        try:
            self.p, self.queue = monitor(self.info, self.output_dir, self.logging_interval)
            self.launched = True
        except Exception as _:
            ex_type, ex_value, tb = sys.exc_info()
            print("Encountered exception when launching tracker.")
            print("".join(traceback.format_tb(tb)))
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Waits for monitoring process to exit.
        Computes and postprocesses more metrics.
        Saves report.
        """
        self.queue.put(STOP_MESSAGE)
        torch.cuda.synchronize()
        self.p.join()

        self.info = load_json(os.path.join(self.output_dir, self.info['tag'] + "_temp.json"))
        os.remove(os.path.join(self.output_dir, self.info['tag'] + "_temp.json"))

        self.info['end_time'] = time.time()
        self.info['{}_total_duration'.format(self.tag)] = self.info['end_time'] - self.info['start_time']

        # if self.num_batches:
        # self.info['per_batch_duration'] = self.info['{}_total_duration'.format(self.tag)] / self.num_batches
        if self.num_examples:
            self.info['per_example_duration'] = self.info['{}_total_duration'.format(self.tag)] / self.num_examples
            self.info['examples_per_second'] = self.num_examples / self.info['{}_total_duration'.format(self.tag)]
        self.info['end_disk_usage'] = shutil.disk_usage(os.path.expanduser('~')).used
        self.info['disk_footprint'] = self.info['end_disk_usage'] - self.info['start_disk_usage']

        for key in self.info['system']:
            if 'gpu_' in key:
                self.info['system'][key]['max_memory_used'] = max(self.info['system'][key]['memory_used'])
        self.info['system']['max_cpu_utilization'] = max(self.info['system']['cpu_utilization'])
        self.info['system']['max_ram_utilization'] = max(self.info['system']['ram_utilization'])
        self.info['system']['max_swap_utilization'] = max(self.info['system']['swap_utilization'])

        save_json(os.path.join(self.output_dir, self.info['tag'] + "_metrics.json"), self.info)
