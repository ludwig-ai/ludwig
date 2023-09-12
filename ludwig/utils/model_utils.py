from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

NUMPY_TO_TORCH_DTYPE = {
    bool: torch.bool,
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def extract_tensors(model: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """Remove the tensors from a PyTorch model, convert them to NumPy arrays, and return the stripped model and
    tensors.

    Reference implementation: https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-
    ray-8be751a6944c  # noqa
    """

    tensors = []
    for _, module in model.named_modules():
        # Store the tensors as numpy arrays in Python dictionaries
        # Delete the same tensors since we no longer need them and we want to reduce memory pressure.
        # This ensures that throughout this process, we keep memory nearly linear w.r.t model parameters.
        params = OrderedDict()
        buffers = OrderedDict()
        for name, param in module.named_parameters(recurse=False):
            params[name] = torch.clone(param).detach().numpy()
            del param
        for name, buf in module.named_buffers(recurse=False):
            buffers[name] = torch.clone(buf).detach().numpy()
            del buf
        tensors.append({"params": params, "buffers": buffers})

    # Strip all tensors and buffers out of the original model.
    for _, module in model.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    return model, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device: torch.device):
    """Restore the tensors that extract_tensors() stripped out of a PyTorch model. This operation is performed in
    place.

    Reference implementation: https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-
    ray-8be751a6944c  # noqa
    """
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device, dtype=NUMPY_TO_TORCH_DTYPE.get(array.dtype))),
            )

        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(
                name,
                torch.as_tensor(array, device=device, dtype=NUMPY_TO_TORCH_DTYPE.get(array.dtype)),
            )


def find_embedding_layer_with_path(module, module_names=[]):
    """Recursively search through a module to find an embedding layer and its module path.

    Returns a tuple containing the embedding layer and its module path.
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, torch.nn.Embedding):
            # If an embedding layer is found, return it along with the module path
            return child_module, ".".join(module_names + [name])
        else:
            # Recursively search in the child module and update the module_names list
            found, path = find_embedding_layer_with_path(child_module, module_names + [name])
            if found is not None:
                return found, path
    return None, None
