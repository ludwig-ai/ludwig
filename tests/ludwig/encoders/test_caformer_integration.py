import torch
from ludwig.encoders.registry import get_encoder_cls

def test_caformer_encoder():
    # Load the CAFormer encoder
    encoder_cls = get_encoder_cls("image", "caformer")
    encoder = encoder_cls(height=224, width=224, num_channels=3, out_channels=512, model_name="caformer_s18")

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Pass the input through the encoder
    output = encoder(dummy_input)

    # Print the output shape
    print("Encoder output shape:", output["encoder_output"].shape)

if __name__ == "__main__":
    test_caformer_encoder()