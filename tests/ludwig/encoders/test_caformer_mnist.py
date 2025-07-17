import pytest
import os
import torch
from ludwig.encoders.image.caformer_encoder import LudwigCAFormerEncoder

@pytest.mark.parametrize("model_name", ["caformer_s18", "caformer_s36", "caformer_m36"])
def test_caformer_on_mnist(model_name):
    # Path to the MNIST dataset
    mnist_path = "tests/ludwig/datasets/mnist"
    assert os.path.exists(mnist_path), f"MNIST dataset not found at {mnist_path}"

    # Load a sample image from the MNIST dataset
    sample_image_path = os.path.join(mnist_path, "sample_image.pt")
    assert os.path.exists(sample_image_path), f"Sample image not found at {sample_image_path}"
    sample_image = torch.load(sample_image_path)  # Assuming the image is saved as a PyTorch tensor

    # Initialize the CAFormer encoder
    encoder = LudwigCAFormerEncoder(
        height=28,  # MNIST images are 28x28
        width=28,
        num_channels=1,  # MNIST images are grayscale
        out_channels=512,
        model_name=model_name,
        use_pretrained=False
    )

    # Pass the sample image through the encoder
    output = encoder(sample_image.unsqueeze(0))  # Add batch dimension
    assert "encoder_output" in output, "Encoder output key missing"
    assert output["encoder_output"].shape == (1, 512), "Unexpected output shape"

    print(f"CAFormer model {model_name} successfully tested on MNIST.")