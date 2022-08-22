from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
from torch._C._autograd import _KinetoEvent
from torch.autograd import DeviceType, profiler_util

from ludwig.constants import LUDWIG_TAG


def initialize_stats_dict(main_function_events: List[profiler_util.FunctionEvent]) -> Dict[str, list]:
    """Initialize dictionary which stores resource usage information per tagged code block.

    :param main_function_events: list of main function events.
    """
    info = {}
    for event_name in [evt.name for evt in main_function_events]:
        info[event_name] = []
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


def get_torch_op_time(events: List[profiler_util.FunctionEvent], attr: str) -> Union[int, float]:
    """Get time torch operators spent executing for a list of events.

    :param events: list of events.
    :param attr: a FunctionEvent attribute. Expecting one of "cpu_time_total", "cuda_time_total".
    """
    if attr not in ["cpu_time_total", "cuda_time_total"]:
        return -1

    total = 0
    for e in events:
        # Possible trace_names are torch ops, or tagged code blocks by LudwigProfiler (which are prepended with LUDWIG_TAG).
        if LUDWIG_TAG not in e.trace_name:
            total += getattr(e, attr)
        else:
            total += get_torch_op_time(e.cpu_children, attr)
    return total


def get_device_run_durations(
    function_event: profiler_util.FunctionEvent, run_usage_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Get CPU and CUDA run durations for an event.

    :param function_event: a function event instance.
    :param run_usage_info: usage info for one execution of the tagged code block.
    """
    run_usage_info["torch_cpu_time"] = get_torch_op_time(function_event.cpu_children, "cpu_time_total")
    run_usage_info["torch_cuda_time"] = get_torch_op_time(function_event.cpu_children, "cuda_time_total")
    return run_usage_info


def get_resource_usage_report(
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
        (evt for evt in main_kineto_events if LUDWIG_TAG in evt.name()), key=lambda x: x.correlation_id()
    )
    main_function_events = sorted((evt for evt in main_function_events if LUDWIG_TAG in evt.name), key=lambda x: x.id)

    for kineto_event, function_event in zip(main_kineto_events, main_function_events):
        run_usage_info = {}
        run_usage_info = get_device_memory_usage(kineto_event, mem_records_acc, run_usage_info)
        run_usage_info = get_device_run_durations(function_event, run_usage_info)
        info[function_event.name].append(run_usage_info)
    return info


def get_all_events(
    kineto_events: List[_KinetoEvent], function_events: profiler_util.EventList
) -> Tuple[List[_KinetoEvent], List[profiler_util.FunctionEvent], List[Any], List[_KinetoEvent]]:
    """Return main Kineto and function events, memory and OOM events for functions/code blocks tagged in
    LudwigProfiler.

    :param kineto_events: list of Kineto Events.
    :param function_events: list of function events.
    """
    # LUDWIG_TAG is prepended to LudwigProfiler tags. This edited tag is passed in to `torch.profiler.record_function`
    # so we can easily retrieve events for code blocks wrapped with LudwigProfiler.
    main_function_events = [evt for evt in function_events if LUDWIG_TAG in evt.name]
    main_kineto_events = [event for event in kineto_events if LUDWIG_TAG in event.name()]
    memory_events = [[event, False] for event in kineto_events if profiler_util.MEMORY_EVENT_NAME in event.name()]
    # profiler_util.OUT_OF_MEMORY_EVENT_NAME seems to only be in newer versions of torch.
    out_of_memory_events = [event for event in kineto_events if "[OutOfMemory]" in event.name()]
    return main_kineto_events, main_function_events, memory_events, out_of_memory_events


def get_metrics_from_torch_profiler(profile: torch.profiler.profiler.profile) -> Dict[str, Any]:
    """Export time and resource usage metrics (CPU and CUDA) from a PyTorch profiler.

    The profiler keeps track of *torch operations* being executed in C++. It keeps track
    of what device they're executed on, their execution time, and memory usage.
    We only track the aforementioned metrics, but the torch profiler can keep track of
    the stack trace, FLOPs, and torch modules. Tracking each additional item adds overhead.

    The torch profiler surfaces these metrics that are tracked under the hood by `libkineto`.
    More on the Kineto project: https://github.com/pytorch/kineto

    :param profile: profiler object that contains all the events that
        were registered during the execution of the wrapped code block.
    """
    # events in both of these lists are in chronological order.
    kineto_events = profile.profiler.kineto_results.events()
    function_events = profile.profiler.function_events
    main_kineto_events, main_function_events, memory_events, _ = get_all_events(kineto_events, function_events)

    assert Counter([event.name for event in main_function_events]) == Counter(
        [event.name() for event in main_kineto_events]
    )
    info = initialize_stats_dict(main_function_events)
    info = get_resource_usage_report(main_kineto_events, main_function_events, memory_events, info)
    return info
