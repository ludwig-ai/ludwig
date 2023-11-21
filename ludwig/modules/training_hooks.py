import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class TrainingHook(ABC):
    """A base class for training hooks in PyTorch.

    This class provides a template for implementing custom training hooks
    that can be activated, deactivated, and maintain a handle to the hook.

    Attributes:
        handle (Optional[torch.utils.hooks.RemovableHandle]): A handle to the
            registered forward hook, initially set to None.

    Methods:
        hook_fn(module: nn.Module, inputs, outputs):
            An abstract method to be implemented by subclasses.
            Defines the custom behavior of the training hook during a forward pass.

        activate_hook(module: nn.Module, hook_fn: Optional[Callable] = None):
            Activates the training hook for a given module.

        deactivate_hook():
            Deactivates and removes the training hook.
    """

    def __init__(self, **kwargs) -> None:
        self._hook_handle = None

    @abstractmethod
    def hook_fn(self, module: torch.nn.Module, inputs: torch.tensor, outputs: torch.tensor) -> torch.tensor:
        """Abstract method to be implemented by subclasses.

        Args:
            module (nn.Module): The PyTorch module for which the hook is activated.
            inputs: The input to the module during the forward pass.
            outputs: The output from the module during the forward pass.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    def activate_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """Activates the training hook for a given module.

        Args:
            module (nn.Module): The PyTorch module for which the hook is activated.
            hook_fn (Optional[Callable]): Custom hook function. If not provided,
                uses the hook_fn defined in the subclass.

        Raises:
            RuntimeError: If the hook function is not provided and the subclass
                does not implement the abstract hook_fn method.
        """
        self._hook_handle = module.register_forward_hook(self.hook_fn)

    def deactivate_hook(self):
        """Deactivates and removes the training hook.

        Raises:
            RuntimeError: If the hook is not activated.
        """
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

        Args:
            module (`torch.nn.Module`):
                The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha`
                to the desired noise alpha value.
            input (`torch.Tensor`):
                The input tensor to the model.
            output (`torch.Tensor`):
                The output tensor of the model (i.e. the embeddings).
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
        """
        from peft import PeftModel

        if isinstance(module, PeftModel):
            embeddings = module.base_model.model.get_input_embeddings()
        else:
            embeddings = module.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        self._hook_handle = embeddings.register_forward_hook(self.hook_fn)

        return module
