import torch
from torch import nn

try:
    from bitsandbytes.functional import dequantize_4bit
    from bitsandbytes.nn.modules import Linear4bit
except ImportError:
    dequantize_4bit = None
    Linear4bit = None

from ludwig.api_annotations import DeveloperAPI


@DeveloperAPI
def linear4bit_to_linear(linear4bit_layer):
    """Converts a Linear4Bit layer to a standard Linear layer by dequantizing the weight values and copying the
    dequantized weights to a new Linear layer.

    Args:
        linear4bit_layer (Linear4bit): The input Linear4Bit layer.

    Returns:
        nn.Linear: A new Linear layer with dequantized weights and biases.
    """
    # Create a new Linear layer with the same shape
    new_linear_layer = nn.Linear(
        linear4bit_layer.in_features,
        linear4bit_layer.out_features,
        bias=linear4bit_layer.bias is not None,
        dtype=torch.float16,
    )

    # Dequantize the weight and bias from the Linear4bit layer and perform an in-place tensor replacement
    # to update the weights and bias in the new Linear layer. This is done to avoid creating a new tensor
    # and copying the data, which is slow.
    new_linear_layer.weight.data.copy_(
        dequantize_4bit(linear4bit_layer.weight.data, linear4bit_layer.weight.quant_state)
    )
    if linear4bit_layer.bias is not None:
        new_linear_layer.bias.data.copy_(linear4bit_layer.bias.data)

    return new_linear_layer


@DeveloperAPI
def convert_quantized_linear_to_linear(module):
    """Recursively converts Linear4Bit layers to standard Linear layers in a given module.

    Args:
        module (nn.Module): The input module containing potentially nested Linear4Bit layers.

    Returns:
        None
    """
    for name, child in module.named_children():
        if isinstance(child, Linear4bit):
            # Replace Linear4Bit layer with a new Linear layer
            setattr(module, name, linear4bit_to_linear(child))
        else:
            # Recursively apply the conversion for nested modules
            convert_quantized_linear_to_linear(child)
