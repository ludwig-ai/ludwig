import torch
from transformers import AutoModelForCausalLM

from ludwig.utils.model_utils import extract_tensors, find_embedding_layer_with_path, replace_tensors

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


# Define a sample module structure for testing
class SampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 20)
        self.rnn = torch.nn.LSTM(20, 30)


def test_find_embedding_layer_with_path_simple():
    # Test case 1: Test the function with a simple module structure
    module = SampleModule()
    embedding_layer, path = find_embedding_layer_with_path(module)
    assert embedding_layer is not None
    assert isinstance(embedding_layer, torch.nn.Embedding)
    assert path == "embedding"


def test_find_embedding_layer_with_path_complex():
    # Test case 2: Test the function with a more complex module structure including AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")

    embedding_layer, path = find_embedding_layer_with_path(model)
    assert embedding_layer is not None
    assert isinstance(embedding_layer, torch.nn.Embedding)
    assert path == "model.embed_tokens"


def test_no_embedding_layer():
    # Test case 3: Embedding layer is not present
    no_embedding_model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))
    embedding_layer, path = find_embedding_layer_with_path(no_embedding_model)
    assert embedding_layer is None
    assert path is None
