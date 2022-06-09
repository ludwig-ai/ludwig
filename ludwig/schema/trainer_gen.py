import textwrap

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


excel = pd.read_excel("sprint.xlsx")
# print(excel)
mask = excel["Class"] == "#/definitions/TrainerConfig"
trainer_excel = excel[mask].iloc[:, 3:-2]
print(trainer_excel.columns)
trainer_excel = trainer_excel.rename(columns=lambda cname: "_".join([w.lower() for w in cname.split(" ")]))
trainer_dict = trainer_excel.set_index("parameter_name").T.to_dict()

# print(trainer_dict)

# Metadata dicts
with open("trainer_metadata.py", "w") as f:
    for param in trainer_dict:
        f.write(f"{param}_metadata = {trainer_dict[param]}\n")

with open("trainer_gen_test.py", "w") as f:
    f.write(
        textwrap.dedent(
            """
    from typing import Optional, Union
    from marshmallow_dataclass import dataclass

    from ludwig.constants import COMBINED, LOSS, TRAINING
    from ludwig.schema import utils as schema_utils
    from ludwig.schema.optimizers import (
        BaseOptimizerConfig,
        GradientClippingConfig,
        GradientClippingDataclassField,
        OptimizerDataclassField,
    )

    @dataclass
    class TrainerConfig(schema_utils.BaseMarshmallowConfig):
    """
        ).expandtabs(4)
    )

    for param in trainer_dict:
        parameter_string = (
            f"\t{param}_metadata: {trainer_dict[param]['Annotation (type)']} = schema_utils.FILLIN(".expandtabs(4)
        )
        default = trainer_dict[param]["Default value"]
        if type(default) == str:
            default = f"'{default}'"
        parameter_string += f"default={default}, "
        allow_none = "'?'"
        # if type(eval(trainer_dict[param]["Annotation (type)"]) == list):
        #     allow_none = "null" in trainer_dict[param]["Annotation (type)"]
        parameter_string += f"allow_none={allow_none}, "
        parameter_string += f"description='{trainer_dict[param]['Description']}', "
        parameter_string += f"metadata={trainer_dict[param]}"
        parameter_string += ")"
        f.write(parameter_string + "\n")
