import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf
import os

def test_hydra_initialization():
    # Set the working directory to the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create a temporary configuration file
    config_dir = "temp_config"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.yaml")
    with open(config_file, "w") as f:
        f.write("defaults: []\n")

    # Initialize Hydra with the relative configuration path
    with initialize(config_path=config_dir):
        cfg = compose(config_name="config")
        assert cfg is not None

    # Clean up the temporary configuration file
    os.remove(config_file)
    os.rmdir(config_dir)