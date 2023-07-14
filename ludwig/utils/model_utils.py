from typing import Dict, List, Tuple

import torch


def extract_tensors(model: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """Remove the tensors from a PyTorch model, convert them to NumPy arrays, and return the stripped model and
    tensors.

    Reference implementation: https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-
    ray-8be751a6944c  # noqa
    """

    tensors = []
    for _, module in model.named_modules():
        # Store the tensors as numpy arrays in Python dictionaries
        # Move the same tensors to a meta device since we no longer need them and we want to reduce memory pressure.
        # This ensures that throughout this process, we keep memory nearly linear w.r.t model parameters.
        params = {}
        buffers = {}
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
            module.register_parameter(name, torch.nn.Parameter(torch.as_tensor(array, device=device)))

        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))
