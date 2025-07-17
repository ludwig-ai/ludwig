import os
import torch
from torchvision import datasets, transforms

def preprocess_mnist():
    # Define the MNIST dataset directory
    mnist_path = "tests/ludwig/datasets/mnist"
    os.makedirs(mnist_path, exist_ok=True)

    # Download and preprocess the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=mnist_path, train=True, download=True, transform=transform)

    # Save a sample image as a PyTorch tensor
    sample_image, _ = mnist_dataset[0]  # Get the first image
    sample_image_path = os.path.join(mnist_path, "sample_image.pt")
    torch.save(sample_image, sample_image_path)
    print(f"Sample image saved at {sample_image_path}")

if __name__ == "__main__":
    preprocess_mnist()