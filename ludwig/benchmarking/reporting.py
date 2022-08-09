import os
from torch.profiler.profiler import profile
from torch.autograd import DeviceType
from torch.autograd.profiler_util import MemRecordsAcc, _format_memory, _format_time
from typing import Any, Dict, Tuple
from statistics import mean

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

def get_devices_usage(p, info):
    kineto_results = p.profiler.kineto_results
    mem_records = [[evt, False] for evt in kineto_results.events() if evt.name() == "[memory]"]
    main_events = [[evt, False] for evt in kineto_results.events() if "ludwig" in evt.name()]
    mem_records_acc = MemRecordsAcc(mem_records)
    for evt in main_events:
        info[evt[0].name()] = {"code_block_tag": evt[0].name()}
        print(evt[0].name())
        cpu_so_far, cuda_so_far = 0, 0
        cpu, cuda = [0], [0]
        for mem_record in mem_records_acc.in_interval(evt[0].start_us(), evt[0].start_us() + evt[0].duration_us()):
            # mem_record[0].device_index(): 255 is CPU, 0, 1, 2, etc. is CUDA.
            if mem_record[0].device_type() in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]:
                cpu.append(cpu_so_far + mem_record[0].nbytes())
                cpu_so_far += mem_record[0].nbytes()
            if mem_record[0].device_type() in [DeviceType.CUDA, DeviceType.HIP]:
                cuda.append(cuda_so_far + mem_record[0].nbytes())
                cuda_so_far += mem_record[0].nbytes()
        print(_format_memory(mean(cpu)), _format_memory(max(cpu)), _format_memory(mean(cuda)), _format_memory(max(cuda)))
        # export graphs of usage data
        # put cuda usage in corresponding cuda device id
        info[evt[0].name()]["average_cpu_memory_usage"] = mean(cpu)
        info[evt[0].name()]["max_cpu_memory_usage"] = max(cpu)
        info[evt[0].name()]["average_cuda_memory_usage"] = mean(cuda)
        info[evt[0].name()]["max_cuda_memory_usage"] = max(cuda)
    return info

def get_device_timing(p, info):
    function_events = p.profiler.function_events
    main_events = [evt for evt in function_events if "ludwig" in evt.name]
    for evt in main_events:
        print(evt.name)
        # evt.self_cpu_time_total
        # evt.cuda_time_total
        # evt.self_cuda_time_total
        # evt.cpu_time_total
        # evt.cpu_time_str
        # evt.cuda_time_str
        # evt.cpu_time_total_str
        # evt.cuda_time_total_str
        # evt.self_cpu_time_total_str
        # evt.self_cuda_time_total_str
        # evt.cpu_time
        # evt.cuda_time
        # I think evt.device_index can give the GPU number
        # name, trace_name, key seem to return the same thing.
    return info


def export_metrics_from_torch_profiler(p: profile, experiment_name: str):
    info = dict()
    info = get_devices_usage(p, info)
    info = get_device_timing(p, info)
    for code_block_tag, report in info.items():
        file_path = os.path.join(os.getcwd(), experiment_name, "metrics_report", "{}.json".format(code_block_tag))
        save_json(file_path, report)
    pass