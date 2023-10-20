import logging
from abc import ABC, abstractmethod
from typing import Callable

import torch
from transformers import AutoConfig

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry

logger = logging.getLogger(__name__)


_neftune_registry = Registry()


@DeveloperAPI
def get_neftune_registry() -> Registry:
    """Retrieve the NEFTune module registry.

    Returns:
        Registry: A registry object for storing NEFTune modules.
    """
    return _neftune_registry


@DeveloperAPI
def register_neftune_architecture(name: str) -> Callable:
    """Decorator for registering custom NEFTune modules.

    This decorator allows you to register custom NEFTune modules to be used with
    different model architectures. The registered modules are identified by their
    model configuration file's "architectures" field.

    Args:
        name (str): Name of the NEFTune module to register. This name should match the
            value in the model configuration file's "architectures" field.

    Returns:
        function: A wrapper function that registers the provided NEFTune module class
            with the specified 'name'.
    """

    def wrap(cls):
        get_neftune_registry()[name] = cls
        return cls

    return wrap


@DeveloperAPI
class NoisedEmbedding(torch.nn.Module):
    """A custom module for adding noise to token embeddings during training.

    This module is designed to modify the behavior of token embeddings by adding noise
    during the training phase while leaving the embeddings unchanged during generation.
    The noise added to embeddings is controlled by the 'noise_alpha' parameter.

    Args:
        orig_embedding (torch.nn.Embedding): The original embedding layer to be modified.
        noise_alpha (int, optional): The magnitude of noise to add during training. Defaults to 5.

    Attributes:
        orig_embedding (torch.nn.Embedding): The original embedding layer.
        noise_alpha (int): The magnitude of noise to be added during training.

    Methods:
        forward(x): The forward method to apply the noised embedding transformation.

    During training, noise is added to the embedding vectors to introduce variability.
    During generation, no noise is added, and the original embeddings are returned.

    For more information, refer to the paper: https://arxiv.org/pdf/2310.05914.pdf
    """

    def __init__(self, orig_embedding: torch.nn.Embedding, noise_alpha: int = 5):
        super().__init__()
        self.orig_embedding = orig_embedding
        self.noise_alpha = noise_alpha

    def forward(self, x):
        if self.training:
            # During training, add noise to the embedding
            embed_init = self.orig_embedding(x)
            dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
            mag_norm = self.noise_alpha / torch.sqrt(dims)
            return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
        else:
            # During generation, don't add noise to the embedding
            return self.orig_embedding(x)


@DeveloperAPI
class BaseArchitectureNEFTune(ABC):
    @abstractmethod
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        """NEFTune method to be implemented by the child class.

        This method should customize the token embedding in the model architecture.
        """
        pass


@DeveloperAPI
@register_neftune_architecture("LlamaForCausalLM")
class LlamaNEFTune(BaseArchitectureNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        orig_embedding = model.model.embed_tokens
        noised_embedding = NoisedEmbedding(orig_embedding, noise_alpha)
        model.model.embed_tokens = noised_embedding
        return model


@DeveloperAPI
@register_neftune_architecture("MistralForCausalLM")
class MistralNEFTune(LlamaNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        return super().NEFTune(model, noise_alpha)


@DeveloperAPI
@register_neftune_architecture("OPTForCausalLM")
class OPTNEFTune(BaseArchitectureNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        orig_embedding = model.model.decoder.embed_tokens
        noised_embedding = NoisedEmbedding(orig_embedding, noise_alpha)
        model.model.decoder.embed_tokens = noised_embedding
        return model


@DeveloperAPI
@register_neftune_architecture("BartForCaualLM")
class BartNEFTune(OPTNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        return super().NEFTune(model, noise_alpha)


@DeveloperAPI
@register_neftune_architecture("BloomForCausalLM")
class BloomNEFTune(BaseArchitectureNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        orig_embedding = model.transformer.word_embeddings
        noised_embedding = NoisedEmbedding(orig_embedding, noise_alpha)
        model.transformer.word_embeddings = noised_embedding
        return model


@DeveloperAPI
@register_neftune_architecture("FalconForCausalLM")
class FalconNEFTune(BloomNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        return super().NEFTune(model, noise_alpha)


@DeveloperAPI
@register_neftune_architecture("GPTJForCausalLM")
class GPTJNEFTune(BaseArchitectureNEFTune):
    def NEFTune(self, model: torch.nn.Module, noise_alpha: int = 5) -> torch.nn.Module:
        orig_embedding = model.transformer.wte
        noised_embedding = NoisedEmbedding(orig_embedding, noise_alpha)
        model.transformer.wte = noised_embedding
        return model


@DeveloperAPI
def NEFTune(model: torch.nn.Module, model_config: AutoConfig, noise_alpha: int = 5) -> torch.nn.Module:
    """Customize token embedding in the model architecture with NEFTune. This is done by adding noise to the
    embedding during training, and not adding noise during generation.

    Paper: https://arxiv.org/pdf/2310.05914.pdf

    Args:
        model (torch.nn.Module): The model to be customized.
        model_config (AutoConfig): The configuration for the model.
        noise_alpha (int, optional): The noise magnitude during embedding token modification.
            Defaults to 5.

    Returns:
        torch.nn.Module: The model with customized token embedding implementation.
    """
    # Check if model architecture is present in the model configuration file
    # If not, sometimes you can find the architecture name in the model class name
    model_architecture = model_config.architectures or model.__class__.__name__ or None
    if model_architecture is None:
        logger.warning(
            "Model architecture not found in model's configuration file. This is required for NEFTune. "
            "Skipping NEFTune."
        )
        return model

    if isinstance(model_architecture, list):
        model_architecture = model_architecture[0]

    if model_architecture not in get_neftune_registry():
        logger.warning(f"NEFTune module for model architecture '{model_architecture}' not found. Skipping NEFTune.")
        return model

    neftune_module = get_neftune_registry()[model_architecture]
    neftune_module().NEFTune(model, noise_alpha)
    return model
