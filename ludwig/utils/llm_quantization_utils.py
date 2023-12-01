from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn.modules import Linear4bit
from torch import nn

from ludwig.api_annotations import DeveloperAPI


@DeveloperAPI
class Linear4BitToLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, linear4bit_layer):
        # Create a new Linear layer with the same shape
        new_linear_layer = nn.Linear(
            linear4bit_layer.in_features, linear4bit_layer.out_features, bias=linear4bit_layer.bias is not None
        )

        # Copy the weight and bias from the Linear4Bit layer to the new Linear layer
        # Dequantize the weight and bias from the Linear4bit layer
        new_linear_layer.weight.data = dequantize_4bit(
            linear4bit_layer.weight.data, linear4bit_layer.weight.quant_state
        )
        if linear4bit_layer.bias is not None:
            new_linear_layer.bias.data = linear4bit_layer.bias.data

        return new_linear_layer


@DeveloperAPI
def convert_linear4bit_to_linear(module):
    for name, child in module.named_children():
        if isinstance(child, Linear4bit):
            # Replace Linear4Bit layer with a new Linear layer
            setattr(module, name, Linear4BitToLinear()(child))
        else:
            # Recursively apply the conversion for nested modules
            convert_linear4bit_to_linear(child)
