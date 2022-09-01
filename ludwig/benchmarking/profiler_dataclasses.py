import dataclasses
from dataclasses import dataclass
from typing import Dict
from ludwig.utils.data_utils import flatten_dict

@dataclass
class DeviceUsageMetrics:
    # Max CUDA memory utilization of the code block.
    max_memory_used: float

    # Average CUDA memory utilization of the code block.
    average_memory_used: float

@dataclass
class SystemResourceMetrics:
    # Name of the code block/function to be profiled.
    code_block_tag: str

    # Name of the CPU that the code ran on.
    cpu_name: str

    # CPU architecture that the code ran on.
    cpu_architecture: str

    # Number of CPUs on the machine.
    num_cpu: int

    # CPU memory available at the beginning of the code block.
    cpu_memory_available: float

    # Ludwig version in the environment.
    ludwig_version: str

    # Total execution time of the code block.
    total_execution_time: float

    # The change in disk memory before and after the code block ran.
    disk_footprint: float

    # Max CPU utilization of the code block.
    max_cpu_utilization: float

    # Max CPU memory (RAM) utilization of the code block.
    max_cpu_memory_usage: float

    # Average CPU utilization of the code block.
    average_cpu_utilization: float

    # Avergae CPU memory (RAM) utilization of the code block.
    average_cpu_memory_usage: float

    # Per device usage. Dictionary containing max and average memory used per device.
    device_usage: Dict[str, DeviceUsageMetrics]

    def to_flat_dict(self):
        nested_dict = dataclasses.asdict(self)
        nested_dict[""] = nested_dict.pop("device_usage")
        return flatten_dict(nested_dict, sep="")


@dataclass
class TorchProfilerMetrics:
    # Time taken by torch ops to execute on the CPU.
    torch_cpu_time: float

    # Time taken by torch ops to execute on CUDA devices.
    torch_cuda_time: float

    # Number of out of memory events.
    num_oom_events: int

    # Per device usage by torch ops. Dictionary containing max and average memory used per device.
    device_usage: Dict[str, DeviceUsageMetrics]

    def to_flat_dict(self):
        nested_dict = dataclasses.asdict(self)
        nested_dict[""] = nested_dict.pop("device_usage")
        return flatten_dict(nested_dict, sep="")
