import torch

from ludwig.utils.model_utils import extract_tensors, replace_tensors

# Define a sample model for testing


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()


def test_extract_tensors():
    # Create a sample model
    model = SampleModel()

    # Call extract_tensors function
    stripped_model, tensors = extract_tensors(model)

    # Assert that the model and tensors are returned
    assert isinstance(stripped_model, torch.nn.Module)
    assert isinstance(tensors, list)

    # Assert that the tensors contain the expected keys
    for tensor_dict in tensors:
        assert "params" in tensor_dict
        assert "buffers" in tensor_dict

    # Assert that all model parameters are set to None
    for module in stripped_model.modules():
        for name, param in module.named_parameters(recurse=False):
            assert param is None

        for name, buf in module.named_buffers(recurse=False):
            assert buf is None


def test_replace_tensors():
    # Create a sample model
    model = SampleModel()

    # Call extract_tensors function to get the tensors
    _, tensors = extract_tensors(model)

    # Create a new device for testing
    device = torch.device("cpu")

    # Call replace_tensors function
    replace_tensors(model, tensors, device)

    # Assert that the tensors are restored
    for module, tensor_dict in zip(model.modules(), tensors):
        for name, array in tensor_dict["params"].items():
            assert name in module._parameters
            assert torch.allclose(module._parameters[name], torch.as_tensor(array, device=device))

        for name, array in tensor_dict["buffers"].items():
            assert name in module._buffers
            assert torch.allclose(module._buffers[name], torch.as_tensor(array, device=device))
