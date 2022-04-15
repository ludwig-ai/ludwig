# !/usr/bin/env python
import json
import os
import subprocess
from pathlib import Path

# Import (unused) modules with marshmallow classes, necessary for generating list of subclasses.
import ludwig.combiners.combiners as lcc  # noqa: F401
import ludwig.marshmallow_utils.test_classes as lut  # noqa: F401
import ludwig.models.trainer as lmt  # noqa: F401
import ludwig.modules.optimization_modules as lmo  # noqa: F401
from ludwig.marshmallow_utils.schema import BaseMarshmallowConfig, get_fully_qualified_class_name

# Helper methods:


def all_subclasses(cls):
    """Returns recursively-generated list of all children classes inheriting from given `cls`."""
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_mclass_paths():
    """Returns a dict of all known marshmallow dataclasses in Ludwig paired with their fully qualified paths within
    the directory."""
    all_mclasses = list(all_subclasses(BaseMarshmallowConfig))
    all_mclasses += [BaseMarshmallowConfig, lcc.CommonTransformerConfig]
    return {cls.__name__: get_fully_qualified_class_name(cls) for cls in all_mclasses}


def prune_dict_except(d, safe_keys):
    """Helper method to delete all keys except `safe_keys` from dict `d`."""
    for key in list(d.keys()):
        if key not in safe_keys:
            del d[key]


def get_pytkdocs_structure_for_path(path: str, docstring_style="restructured-text"):
    """Runs pytkdocs in a subprocess and returns the parsed structure of the object at the given path with the
    given documentation style."""
    pytkdocs_subprocess = subprocess.Popen(
        ["pytkdocs"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    input_json = json.dumps({"objects": [{"path": path, "docstring_style": docstring_style}]})
    pytkdocs_output = pytkdocs_subprocess.communicate(input=input_json.encode())[0]
    return json.loads(pytkdocs_output.decode())


# Extraction methods:


def prune_pytorch_structures(opt_struct):
    """Prunes the given torch optimizer struct of unnecessary information."""
    torch_keywords = ["name", "path", "relative_path", "docstring", "docstring_sections"]
    prune_dict_except(opt_struct, torch_keywords)
    # Prune docstring_sections:
    sections = opt_struct["docstring_sections"]
    save_index = list(map(lambda s: "type" in s and s["type"] == "parameters", sections)).index(True)
    opt_struct["docstring_sections"] = [sections[save_index]]


def extract_pytorch_structures():
    """Extracts and saves the parsed structure of all pytorch classes referenced in
    `ludwig.modules.optimization_modules.optimizer_registry` as JSON files under
    `ludwig/marshmallow_utils/generated/torch/`."""
    for opt in lmo.optimizer_registry:
        # Get the torch class:
        optimizer_class = lmo.optimizer_registry[opt][0]

        # Parse and clean the class structure:
        path = get_fully_qualified_class_name(optimizer_class)
        opt_struct = get_pytkdocs_structure_for_path(path, "google")["objects"][0]
        prune_pytorch_structures(opt_struct)

        # Write it to a file:
        parent_dir = str(Path(__file__).parent.parent)
        filename = (
            os.path.join(parent_dir, "ludwig/marshmallow_utils/generated/torch/", optimizer_class.__name__) + ".json"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            json.dump(
                opt_struct,
                outfile,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
            )
            outfile.write("\n")


def prune_ludwig_structures(opt_struct):
    """Prunes the given Ludwig class struct of unnecessary information."""
    # Prune top-level:
    toplevel_keywords = ["name", "docstring", "attributes", "bases", "relative_path", "path", "children"]
    prune_dict_except(opt_struct, toplevel_keywords)

    # Prune children to just the (non-class/method) attributes and init:
    init_name = ".".join([opt_struct["path"], "__init__"])
    attrs = opt_struct["attributes"] + [init_name]
    children = opt_struct["children"]
    prune_dict_except(children, attrs)

    # Prune each attribute of unnecessary information:
    attr_keywords = ["docstring", "name", "path", "relative_file_path", "type", "signature"]
    for a in attrs:
        if a in children:
            prune_dict_except(children[a], attr_keywords)


def extract_marshmallow_structures():
    """Extracts and saves the parsed structure of all known marshmallow dataclasses referenced throughout Ludwig as
    JSON files under `ludwig/marshmallow_utils/generated/`."""
    mclass_paths = get_mclass_paths()
    for cls_name, path in mclass_paths.items():
        # Parse and clean the class structure:
        mclass = get_pytkdocs_structure_for_path(path)["objects"][0]
        prune_ludwig_structures(mclass)

        # Write it to a file:
        parent_dir = str(Path(__file__).parent.parent)
        filename = os.path.join(parent_dir, "ludwig/marshmallow_utils/generated/", cls_name) + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            json.dump(mclass, outfile, indent=4, sort_keys=True, separators=(",", ": "))
            outfile.write("\n")


def main():
    """Simple runner for marshmallow dataclass extraction."""
    extract_pytorch_structures()
    extract_marshmallow_structures()


if __name__ == "__main__":
    main()
