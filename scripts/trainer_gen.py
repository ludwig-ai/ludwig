import os

# import textwrap
from ast import literal_eval
from pathlib import Path

# import numpy as np
import pandas as pd


class CodeBlock:
    def __init__(self, head, block):
        self.head = head
        self.block = block

    def __str__(self, indent=""):
        result = indent + self.head + ":\n"
        indent += "    "
        for block in self.block:
            if isinstance(block, CodeBlock):
                result += block.__str__(indent)
            else:
                result += indent + block + "\n"
        return result


excel = pd.read_excel("sprint5.xlsx")
excel.fillna("None", inplace=True)
print(excel["parameter_path"])
mask = excel["class"] == "#/definitions/TrainerConfig"
trainer_excel = excel[mask].iloc[:, 3:-2]
trainer_excel["allow_none"] = trainer_excel["allow_none"].astype(bool)
print(trainer_excel["allow_none"])
# print(trainer_excel["parameter_path"])

# trainer_excel = trainer_excel.rename(columns=lambda cname: "_".join([w.lower() for w in cname.split(" ")]))
trainer_dict = trainer_excel.set_index("parameter_name").T.to_dict()

# print(trainer_dict)


def safe_eval(expr):
    try:
        return literal_eval(expr)
    except Exception:
        return expr


# Metadata dicts
print(__file__)
print(Path(__file__).parent)
print(os.path.dirname(__file__))
print(os.path.dirname(os.path.dirname(__file__)))
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ludwig/schema/metadata/trainer.py")
with open(path, "w") as f:
    for param in trainer_dict:
        # Clear NaNs:
        metadata_dict = {k: None if v == "None" else safe_eval(v) for k, v in trainer_dict[param].items()}

        # Convert literal types:
        f.write(f"{param}_metadata = {metadata_dict}\n")

# with open("trainer_gen_test.py", "w") as f:
#     f.write(
#         textwrap.dedent(
#             """
#     from typing import Optional, Union
#     from marshmallow_dataclass import dataclass

#     from ludwig.constants import COMBINED, LOSS, TRAINING
#     from ludwig.schema import utils as schema_utils
#     from ludwig.schema.optimizers import (
#         BaseOptimizerConfig,
#         GradientClippingConfig,
#         GradientClippingDataclassField,
#         OptimizerDataclassField,
#     )

#     @dataclass
#     class TrainerConfig(schema_utils.BaseMarshmallowConfig):
#     """
#         ).expandtabs(4)
#     )

#     for param in trainer_dict:
#         parameter_string = (
#             f"\t{param}_metadata: {trainer_dict[param]['Annotation (type)']} = schema_utils.FILLIN(".expandtabs(4)
#         )
#         default = trainer_dict[param]["Default value"]
#         if type(default) == str:
#             default = f"'{default}'"
#         parameter_string += f"default={default}, "
#         allow_none = "'?'"
#         # if type(eval(trainer_dict[param]["Annotation (type)"]) == list):
#         #     allow_none = "null" in trainer_dict[param]["Annotation (type)"]
#         parameter_string += f"allow_none={allow_none}, "
#         parameter_string += f"description='{trainer_dict[param]['Description']}', "
#         parameter_string += f"metadata={trainer_dict[param]}"
#         parameter_string += ")"
#         f.write(parameter_string + "\n")
