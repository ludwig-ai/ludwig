import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class TrainingHook(ABC):
    """A base class for training hooks in PyTorch.

    This class provides a template for implementing custom training hooks
    that can be activated, deactivated, and maintain a handle to the hook.

    Attributes:
        _hook_handle (Optional[torch.utils.hooks.RemovableHandle]): A handle to the
            registered forward hook, initially set to None.
    """

    def __init__(self, **kwargs) -> None:
        self._hook_handle = None

    @abstractmethod
    def hook_fn(self, module: torch.nn.Module, inputs: torch.tensor, outputs: torch.tensor) -> torch.tensor:
        """Abstract method to be implemented by subclasses. This is the method that defines the custom behavior of
        the training hook during a forward pass for the specified module.

        Args:
            module (nn.Module): The PyTorch module for which the hook is activated.
            inputs (torch.Tensor): The input to the module during the forward pass.
            outputs (torch.Tensor): The output from the module during the forward pass.

        Returns:
            torch.Tensor: The output tensor from the module.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    def activate_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """Activates the training hook for a given module.

        Args:
            module (nn.Module): The PyTorch module for which the hook is activated.

        Returns:
            nn.Module: The input module with the training hook activated.
        """
        self._hook_handle = module.register_forward_hook(self.hook_fn)
        return module

    def deactivate_hook(self):
        """Deactivates and removes the training hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class NEFTuneHook(TrainingHook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.neftune_noise_alpha = kwargs.get("neftune_noise_alpha")

    def hook_fn(self, module: torch.nn.Module, input: torch.tensor, output: torch.tensor) -> torch.tensor:
        """Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
        torch.nn. Embedding layers. This method is slightly adapted from the original source code that can be found
        here: https://github.com/neelsjain/NEFTune.

        The input tensor is ignored since the noise is added to the output of the embedding layer.

        Returns:
            torch.Tensor: The output tensor from the module.
        """
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output

    def activate_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """Activates the neftune as presented in this code and paper:

        Code: https://github.com/neelsjain/NEFTune
        Paper: https://arxiv.org/abs/2310.05914

        Args:
            module (nn.Module): The PyTorch module for which the hook is activated.

        Returns:
            nn.Module: The input module with the training hook activated.
        """
        from peft import PeftModel

        if isinstance(module, PeftModel):
            embeddings = module.base_model.model.get_input_embeddings()
        else:
            embeddings = module.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        self._hook_handle = embeddings.register_forward_hook(self.hook_fn)

        return module
