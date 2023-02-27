import copy
import random
from collections import deque
from typing import Any, Deque, Dict, List, Tuple, Union

from ludwig.constants import SEQUENCE, TEXT, TIMESERIES
from ludwig.types import ModelConfigDict
from ludwig.utils.misc_utils import merge_dict


def explore_properties(properties: Dict[str, Any], parent_key: str, dq: Deque[Tuple], only_include=[]) -> Deque[Tuple]:
    """Recursively explores the `properties` part of any subsection of the schema.

    Params:
        properties: any properties section of the schema.
        parent_key: parent dictionary keys up to the current property dictionary (e.g. defaults.number.preprocessing)
        dq: dequeue data structure that stores tuples of (config_options, fully_explored). config_options is
            a dictionary containing the flat config parameters and some values (as a list) to explore following
            the rules set by the schema about this parameter. fully_explored is a boolean value indicating that
            all subsections of the properties dictionary have been explored.
        only_include: list of top level keys of the properties dictionary to skip.
    """
    # processed_dq will contain complete config options with all the parameters in the properties dictionary
    # dq will contain configs options that are still being completed.
    processed_dq = deque()
    while dq and not dq[0][1]:
        for key in properties:
            if only_include and key not in only_include:
                continue

            key_so_far = parent_key + "." + key if parent_key else key
            config_options, _ = dq.popleft()
            item = properties[key]

            if "properties" in item and "allOf" in item:
                for child_item in item["allOf"]:
                    expanded_config_options_dq = explore_from_all_of(
                        config_options=copy.deepcopy(config_options), item=child_item, key_so_far=key_so_far
                    )
                    # add returned child config options to the deque to be processed.
                    dq.extend(expanded_config_options_dq)

            elif "properties" in item and "allOf" not in item:
                child_properties = item["properties"]
                # a new dequeue to be passed to explore parameters from
                raw_entry = deque([(copy.deepcopy(config_options), False)])
                child_config_options_dq = explore_properties(child_properties, key_so_far, raw_entry)
                merged_config_options_dq = merge_dq(config_options, child_config_options_dq)
                # add returned config options to the deque to be processed.
                dq.extend(merged_config_options_dq)

            else:
                # this is the base case.
                if "oneOf" in item:
                    temp = []
                    for elem in item["oneOf"]:
                        temp += get_potential_values(elem)
                    config_options[key_so_far] = temp
                else:
                    config_options[key_so_far] = get_potential_values(item)

                # for config parameters that are internal or for which we can't infer suggested values to explore
                # e.g. parameters of type array, object, string (in some cases), etc.
                if len(config_options[key_so_far]) == 0:
                    del config_options[key_so_far]

                # add config_options back to queue. fully_explored = False because we still didn't finish
                # exploring all the keys in the properties dictionary.
                dq.appendleft((config_options, False))

        # at this point, we finished exploring all keys of the properties dictionary. Add all config options
        # to the processed queue.
        while dq:
            config_options, _ = dq.popleft()
            processed_dq.append((config_options, True))

    return processed_dq


def explore_from_all_of(config_options: Dict[str, Any], item: Dict[str, Any], key_so_far: str):
    """Takes a child of `allOf` and calls `explore_properties` on it."""
    for key in item["if"]["properties"]:
        config_options[key_so_far + "." + key] = item["if"]["properties"][key]["const"]
    properties = item["then"]["properties"]
    raw_entry = deque([(copy.deepcopy(config_options), False)])
    return explore_properties(properties, parent_key=key_so_far, dq=raw_entry)


def get_potential_values(item: Dict[str, Any]):
    """Returns a list of values to explore for a config parameter.

    Param:
        item: config parameter-specific dictionary. Considered as a leaf in the schema. Contains type, default, and
            parameter metadata, etc.
    """
    temp = []
    if isinstance(item["type"], list):
        for property_type in item["type"]:
            temp += handle_property_type(property_type, item)
    else:
        temp += handle_property_type(item["type"], item)
    unique_temp = []
    for temp_item in temp:
        if temp_item not in unique_temp:
            unique_temp.append(temp_item)
    return unique_temp


def handle_property_type(property_type, item):
    # don't explore internal only parameters.
    if "parameter_metadata" in item and item["parameter_metadata"] and item["parameter_metadata"]["internal_only"]:
        return []
    # don't explore parameters that have priority less than HIGH.
    if (
        "parameter_metadata" in item
        and item["parameter_metadata"]
        and item["parameter_metadata"]["expected_impact"] < 3
    ):
        return []

    if property_type == "number":
        return explore_number(item)
    elif property_type == "integer":
        return explore_integer(item)
    elif property_type == "string":
        return explore_string(item)
    elif property_type == "boolean":
        return explore_boolean()
    elif property_type == "null":
        return explore_null()
    elif property_type == "array":
        return explore_array(item)
    else:
        return []


def explore_array(item):
    candidates = []
    if "default" in item and item["default"]:
        candidates.append(item["default"])

    item_choices = []
    maxlen = 0

    # In the case where the length of the array isn't defined.
    if not isinstance(item["items"], list):
        return []

    for it in item["items"]:
        choices = handle_property_type(it["type"], it)
        maxlen = max(maxlen, len(choices))
        item_choices.append(choices)

    # pad to same length
    for i in range(len(item_choices)):
        item_choices[i] = maxlen * item_choices[i]
        item_choices[i] = item_choices[i][:maxlen]

    merged = list(zip(*item_choices)) + candidates
    return [list(tup) for tup in merged]


def explore_number(item):
    # add min and max rules
    minimum, maximum = 0, 1
    if "default" not in item or item["default"] is None:
        candidates = []
    else:
        candidates = [1, 2, item["default"], 2 * (item["default"] + 1), item["default"] // 2, -1 * item["default"]]

    if "minimum" in item:
        minimum = item["minimum"]
        candidates = [num for num in candidates if num > minimum]
    if "maximum" in item:
        maximum = item["maximum"]
        candidates = [num for num in candidates if num < maximum]
    return candidates + [random.random() * 0.99 * maximum]


def explore_integer(item):
    # add min and max rules
    minimum, maximum = 0, 10

    if "default" not in item or item["default"] is None:
        candidates = []
    else:
        candidates = [item["default"], 2 * (item["default"] + 1), item["default"] // 2, -1 * item["default"]]

    if "minimum" in item:
        minimum = item["minimum"]
        candidates = [num for num in candidates if num >= item["minimum"]]
    if "maximum" in item:
        maximum = item["maximum"]
        candidates = [num for num in candidates if num <= item["maximum"]]

    return candidates + [random.randint(minimum, maximum)]


def explore_string(item):
    if "enum" in item:
        return item["enum"]
    return [item["default"]]


def explore_boolean():
    return [True, False]


def explore_null():
    return [None]


def merge_dq(config_options: Dict[str, Any], child_config_options_dq: Deque[Tuple]):
    """Merge config_options with the child_config_options in the dq."""
    dq = deque()
    while child_config_options_dq:
        child_config_options, visited = child_config_options_dq.popleft()
        cfg = merge_dict(child_config_options, config_options)
        dq.append((cfg, visited))
    return dq


def generate_possible_configs(config_options: Dict[str, Any]):
    """Generate exhaustive configs from config_options.

    This function does not take a cross product of all the options for all the config parameters. It selects parameter
    values independently from each other.
    """
    num_configs = 1
    for key in config_options:
        if isinstance(config_options[key], list):
            num_configs = max(num_configs, len(config_options[key]))
            config_options[key] = deque(config_options[key])

    for _ in range(num_configs):
        config = {}
        for key in config_options:
            if config_options[key] and not isinstance(config_options[key], str):
                config[key] = config_options[key].popleft()
            elif isinstance(config_options[key], str):
                config[key] = config_options[key]
        yield config


def create_nested_dict(flat_dict: Dict[str, Union[float, str]]) -> Dict[str, Any]:
    """Generate a nested dict out of a flat dict whose keys are delimited by a delimiter character.

    Params:
        flat_dict: potential generated baseline config.
    """

    def to_nested_format(key: str, value: Union[str, int, float], delimiter: str = ".") -> Dict[str, Any]:
        # https://stackoverflow.com/a/40401961
        split_key = key.split(delimiter)
        for key in reversed(split_key):
            value = {key: value}
        return value

    config = {}
    for key in flat_dict:
        config = merge_dict(config, to_nested_format(key, copy.deepcopy(flat_dict[key])))
    return config


def combine_configs(explored, config, dataset_name) -> List[Tuple[ModelConfigDict, str]]:
    ret = []
    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            default_config = create_nested_dict(default_config)
            merged_config = merge_dict(copy.deepcopy(config), default_config)
            ret.append((merged_config, dataset_name))
    return ret


def combine_configs_for_comparator_combiner(explored, config, dataset_name) -> List[Tuple[ModelConfigDict, str]]:
    ret = []
    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            default_config = create_nested_dict(default_config)
            merged_config = merge_dict(copy.deepcopy(config), default_config)

            # create two random lists of random lengths for entity1 and entity2
            num_entities = random.randint(2, len(config["input_features"]))
            entity_names = [feature["name"] for feature in random.sample(config["input_features"], num_entities)]
            entity_1_size = random.randint(1, num_entities - 1)
            merged_config["combiner"]["entity_1"] = entity_names[:entity_1_size]
            merged_config["combiner"]["entity_2"] = entity_names[entity_1_size:]
            ret.append((merged_config, dataset_name))
    return ret


def combine_configs_for_sequence_combiner(explored, config, dataset_name) -> List[Tuple[ModelConfigDict, str]]:
    ret = []
    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            default_config = create_nested_dict(default_config)
            merged_config = merge_dict(copy.deepcopy(config), default_config)
            for i in range(len(merged_config["input_features"])):
                if merged_config["input_features"][i]["type"] in {SEQUENCE, TEXT, TIMESERIES}:
                    merged_config["input_features"][0]["encoder"] = {"type": "embed", "reduce_output": None}
            ret.append((merged_config, dataset_name))
    return ret
