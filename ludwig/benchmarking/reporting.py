import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Tuple

from torch.autograd import DeviceType
from torch.autograd.profiler_util import _format_memory, MemRecordsAcc
from torch.profiler.profiler import profile

from ludwig.constants import CACHE, EVAL_TAG, EXPERIMENT_RUN, TRAIN_TAG
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.misc_utils import merge_dict


def create_metrics_report(experiment_name: str) -> Tuple[Dict[str, Any], str]:
    """Compiles performance and non-performance metrics.

    `experiment_name`: name referring to the experiment.
    Returns a full report and the path where it's saved.
    """
    full_report = dict()
    os.makedirs(os.path.join(os.getcwd(), experiment_name, "metrics_report"), exist_ok=True)
    for tag in [TRAIN_TAG, EVAL_TAG]:
        if tag == TRAIN_TAG:
            resource_usage_path = os.path.join(os.getcwd(), experiment_name, CACHE, "train_resource_usage_metrics.json")
            performance_path = os.path.join(os.getcwd(), experiment_name, EXPERIMENT_RUN, "training_statistics.json")
        elif tag == EVAL_TAG:
            resource_usage_path = os.path.join(
                os.getcwd(), experiment_name, CACHE, "evaluate_resource_usage_metrics.json"
            )
            performance_path = os.path.join(os.getcwd(), experiment_name, EXPERIMENT_RUN, "test_statistics.json")
        else:
            raise ValueError("Tag unrecognized. Please choose 'train' or 'evaluate'.")

        resource_usage_metrics = load_json(resource_usage_path)
        performance_metrics = load_json(performance_path)
        full_report[tag] = merge_dict(performance_metrics, resource_usage_metrics)

    merged_file_path = os.path.join(os.getcwd(), experiment_name, "metrics_report", "{}.json".format("full_report"))
    save_json(merged_file_path, full_report)
    return full_report, merged_file_path


def initialize_stats_dict(function_events):
    """Initialize dictionary which stores resource usage information per tagged code block."""
    info = dict()
    for event_name in [evt.name for evt in function_events if "ludwig" in evt.name]:
        info[event_name] = {"code_block_tag": event_name}
        info[event_name]["runs"] = []
    return info


def get_memory_details(event):
    """Get device name and number of bytes (de)allocated during an event."""
    if event.device_type() in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]:
        return "cpu", event.nbytes()
    elif event.device_type() in [DeviceType.CUDA, DeviceType.HIP]:
        return f"cuda_{event.device_index()}", event.nbytes()


def get_devices_usage(kineto_event, mem_records_acc, run_usage_info):
    """Get CPU and CUDA memory usage for and event."""
    memory_so_far = defaultdict(int)
    memory_lists = defaultdict(list)
    for mem_record in mem_records_acc.in_interval(
            kineto_event.start_us(), kineto_event.start_us() + kineto_event.duration_us()
    ):
        device, nbytes = get_memory_details(mem_record[0])
        memory_so_far[device] += nbytes
        memory_lists[device].append(memory_so_far[device])
    for device in memory_lists:
        memory_lists[device].append(0)  # just in case we have an empty list
        run_usage_info[f"average_{device}_memory_usage"] = mean(memory_lists[device])
        run_usage_info[f"max_{device}_memory_usage"] = max(memory_lists[device])
    return run_usage_info


def get_device_timing(function_event, run_usage_info):
    """Get CPU and CUDA run durations for an event."""
    run_usage_info["self_cpu_time_total"] = function_event.self_cpu_time_total
    run_usage_info["cuda_time_total"] = function_event.cuda_time_total
    run_usage_info["self_cuda_time_total"] = function_event.self_cuda_time_total
    run_usage_info["cpu_time_total"] = function_event.cpu_time_total
    run_usage_info["cpu_time"] = function_event.cpu_time
    run_usage_info["cuda_time"] = function_event.cuda_time
    return run_usage_info


def get_resource_usage_report(main_kineto_events, main_function_events, memory_events, info):
    """Get relevant information from Kineto events and function events exported by the profiler."""
    mem_records_acc = MemRecordsAcc(memory_events)
    main_kineto_events = sorted((evt for evt in main_kineto_events if "ludwig" in evt.name()),
                                key=lambda x: x.correlation_id())
    main_function_events = sorted((evt for evt in main_function_events if "ludwig" in evt.name), key=lambda x: x.id)
    assert [evt.id for evt in main_function_events] == [evt.correlation_id() for evt in main_kineto_events]
    assert [evt.name for evt in main_function_events] == [evt.name() for evt in main_kineto_events]
    for kineto_event, function_event in zip(main_kineto_events, main_function_events):
        run_usage_info = {}
        run_usage_info = get_devices_usage(kineto_event, mem_records_acc, run_usage_info)
        run_usage_info = get_device_timing(function_event, run_usage_info)
        info[function_event.name]["runs"].append(run_usage_info)
    return info


def get_all_events(kineto_events, function_events):
    main_function_events = [evt for evt in function_events if "ludwig" in evt.name]
    main_kineto_events = [event for event in kineto_events if "ludwig" in event.name()]
    memory_events = [[event, False] for event in kineto_events if "[memory]" in event.name()]
    return main_kineto_events, main_function_events, memory_events


def export_metrics_from_torch_profiler(p: profile, experiment_name: str):
    """Export time and resource usage metrics (CPU and CUDA) from a PyTorch profiler."""
    # events in both of these lists are in chronological order.
    kineto_events = p.profiler.kineto_results.events()
    function_events = p.profiler.function_events
    main_kineto_events, main_function_events, memory_events = get_all_events(kineto_events, function_events)

    assert Counter([event.name for event in main_function_events]) == Counter(
        [event.name() for event in main_kineto_events])

    info = initialize_stats_dict(function_events)
    info = get_resource_usage_report(main_kineto_events, main_function_events, memory_events, info)
    for code_block_tag, report in info.items():
        os.makedirs(os.path.join(os.getcwd(), experiment_name, "metrics_report"), exist_ok=True)
        file_path = os.path.join(
            os.getcwd(), experiment_name, "metrics_report", f"{code_block_tag}_resource_usage.json"
        )
        save_json(file_path, report)
