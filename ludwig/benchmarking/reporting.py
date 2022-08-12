import os
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch._C._autograd import _KinetoEvent
from torch.autograd import DeviceType, profiler_util

from ludwig.constants import CACHE, EVAL_TAG, EXPERIMENT_RUN, TRAIN_TAG
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.misc_utils import merge_dict


def create_metrics_report(experiment_name: str) -> Tuple[Dict[str, Any], str]:
    """Compiles performance and non-performance metrics.

    :param experiment_name: name referring to the experiment.
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


def initialize_stats_dict(main_function_events: List[profiler_util.FunctionEvent]) -> Dict[str, Any]:
    """Initialize dictionary which stores resource usage information per tagged code block.

    :param main_function_events: list of main function events.
    """
    info = dict()
    for event_name in [evt.name for evt in main_function_events]:
        info[event_name] = {"code_block_tag": event_name}
        info[event_name]["runs"] = []
    return info


def get_memory_details(kineto_event: _KinetoEvent) -> Tuple[str, int]:
    """Get device name and number of bytes (de)allocated during an event.

    :param kineto_event: a Kineto event instance.
    """
    if kineto_event.device_type() in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]:
        return "cpu", kineto_event.nbytes()
    elif kineto_event.device_type() in [DeviceType.CUDA, DeviceType.HIP]:
        return f"cuda_{kineto_event.device_index()}", kineto_event.nbytes()
    else:
        raise ValueError(f"Device {kineto_event.device_type()} is not valid.")


def get_device_memory_usage(
    kineto_event: _KinetoEvent, mem_records_acc: profiler_util.MemRecordsAcc, run_usage_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Get CPU and CUDA memory usage for an event.

    :param kineto_event: a Kineto event instance.
    :param mem_records_acc: memory events wrapped in a data structure for fast lookup.
    :param run_usage_info: usage info for one execution of the tagged code block.
    """
    records_in_interval = mem_records_acc.in_interval(
        kineto_event.start_us(), kineto_event.start_us() + kineto_event.duration_us()
    )
    memory_so_far = defaultdict(int)
    count_so_far = defaultdict(int)
    average_so_far = defaultdict(float)
    max_so_far = defaultdict(int)

    for mem_record in records_in_interval:
        device, nbytes = get_memory_details(mem_record[0])
        memory_so_far[device] += nbytes
        max_so_far[device] = max(max_so_far[device], memory_so_far[device])
        average_so_far[device] = (memory_so_far[device] + (average_so_far[device] * count_so_far[device])) / (
            count_so_far[device] + 1
        )
        count_so_far[device] += 1
    for device in count_so_far:
        run_usage_info[f"torch_average_{device}_memory_usage"] = average_so_far[device]
        run_usage_info[f"torch_max_{device}_memory_usage"] = max_so_far[device]
    return run_usage_info


def get_device_run_durations(
    function_event: profiler_util.FunctionEvent, run_usage_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Get CPU and CUDA run durations for an event.

    :param function_event: a function event instance.
    :param run_usage_info: usage info for one execution of the tagged code block.
    """
    run_usage_info["torch_self_cpu_time_total"] = function_event.self_cpu_time_total
    run_usage_info["torch_cuda_time_total"] = function_event.cuda_time_total
    run_usage_info["torch_self_cuda_time_total"] = function_event.self_cuda_time_total
    run_usage_info["torch_cpu_time_total"] = function_event.cpu_time_total
    run_usage_info["torch_cpu_time"] = function_event.cpu_time
    run_usage_info["torch_cuda_time"] = function_event.cuda_time
    return run_usage_info


def get_resource_usage_report(
    tags: set,
    main_kineto_events: List[_KinetoEvent],
    main_function_events: List[profiler_util.FunctionEvent],
    memory_events: List[Any],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Get relevant information from Kineto events and function events exported by the profiler.

    :param main_kineto_events: list of main Kineto events.
    :param main_function_events: list of main function events.
    :param memory_events: list of memory events.
    :param info: dictionary used to record resource usage metrics.
    """
    mem_records_acc = profiler_util.MemRecordsAcc(memory_events)
    main_kineto_events = sorted(
        (evt for evt in main_kineto_events if evt.name() in tags), key=lambda x: x.correlation_id()
    )
    main_function_events = sorted((evt for evt in main_function_events if evt.name in tags), key=lambda x: x.id)
    assert [evt.id for evt in main_function_events] == [evt.correlation_id() for evt in main_kineto_events]
    assert [evt.name for evt in main_function_events] == [evt.name() for evt in main_kineto_events]
    for kineto_event, function_event in zip(main_kineto_events, main_function_events):
        run_usage_info = {}
        run_usage_info = get_device_memory_usage(kineto_event, mem_records_acc, run_usage_info)
        run_usage_info = get_device_run_durations(function_event, run_usage_info)
        info[function_event.name]["runs"].append(run_usage_info)
    return info


def get_all_events(
    tags: set, kineto_events: List[_KinetoEvent], function_events: profiler_util.EventList
) -> Tuple[List[_KinetoEvent], List[profiler_util.FunctionEvent], List[Any], List[_KinetoEvent]]:
    """Return main Kineto and function events (tagged with "ludwig.*"), memory and out of memory events.

    :param tags: set of code block/function tags we report resource usage metrics for.
    :param kineto_events: list of Kineto Events.
    :param function_events: list of function events.
    """
    main_function_events = [evt for evt in function_events if evt.name in tags]
    main_kineto_events = [event for event in kineto_events if event.name() in tags]
    memory_events = [[event, False] for event in kineto_events if profiler_util.MEMORY_EVENT_NAME in event.name()]
    # profiler_util.OUT_OF_MEMORY_EVENT_NAME seems to only be in newer versions of torch.
    out_of_memory_events = [event for event in kineto_events if "[OutOfMemory]" in event.name()]
    return main_kineto_events, main_function_events, memory_events, out_of_memory_events


def export_metrics_from_torch_profiler(tags: list, profile: torch.profiler.profiler.profile, output_dir: str):
    """Export time and resource usage metrics (CPU and CUDA) from a PyTorch profiler.

    The profiler keeps track of *torch operations* being executed in C++. It keeps track
    of what device they're executed on, their execution time, and memory usage.
    We only track the aforementioned metrics, but the torch profiler can keep track of
    the stack trace, FLOPs, and torch modules. Tracking each additional item adds overhead.

    The torch profiler surfaces these metrics that are tracked under the hood by `libkineto`.
    More on the Kineto project: https://github.com/pytorch/kineto

    :param tags: list of code block/function tags we report resource usage metrics for.
    :param profile: profiler object that contains all the events that
        were registered during the execution of the wrapped code block.
    :param output_dir: a tag for the experiment.
    """
    # events in both of these lists are in chronological order.
    kineto_events = profile.profiler.kineto_results.events()
    function_events = profile.profiler.function_events
    tags = set(tags)
    main_kineto_events, main_function_events, memory_events, _ = get_all_events(tags, kineto_events, function_events)

    assert Counter([event.name for event in main_function_events]) == Counter(
        [event.name() for event in main_kineto_events]
    )

    info = initialize_stats_dict(main_function_events)
    info = get_resource_usage_report(tags, main_kineto_events, main_function_events, memory_events, info)
    for code_block_tag, report in info.items():
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{code_block_tag}_resource_usage.json")
        save_json(file_path, report)
        logging.info(f"exported to {file_path}")
    return info
