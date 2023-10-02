import logging
from typing import Callable, Tuple, Union

import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)


class ParameterUpdateError(Exception):
    pass


def check_module_parameters_updated(
    module: LudwigModule,
    module_input_args: Tuple,
    module_target: torch.Tensor,
    loss_function: Union[Callable, None] = None,
    max_steps: int = 1,
    learning_rate: float = 0.001,
) -> Tuple:
    """
    Reports on the number of parameters in a Ludwig component and their update status.
    Args:
        module: (LudwigModel) model to be tested.
        module_input_args: (tuple) input for model
        module_target: (Tensor) target values for computing loss and parameter updates
        loss_function: (None or Callable) Optional for module specific loss calculation
        max_steps: (int, default=1) maximum number of steps allowed to test for parameter
            updates.
        learning_rate: (float, default=0.001) learning rate for the optimizer

    Returns: Tuple(frozen_parameters, trainable_parameters, parameters_updated, not_updated)
        frozen_parameters: count of frozen parameters
        trainable_parameters: count of trainable parameters
        parameters_updated: count of trainable parameters that were updated
        not_updated: list of parameters that were not updated

    """
    # setup
    if loss_function is None:
        loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    module.train(True)

    target_tensor = module_target

    trainable_parameter_list = []
    frozen_parameter_list = []
    parameter_updated = []
    parameters_not_updated = []
    for step in range(max_steps):
        # make pass through model
        module_output = module(*module_input_args)

        # check for any frozen parameters
        frozen_parameter_list = []
        trainable_parameter_list = []
        for p in module.named_parameters():
            if p[1].requires_grad:
                trainable_parameter_list.append(p)
            else:
                frozen_parameter_list.append(p)

        # check parameter updates only if there are some unfrozen parameters
        if len(trainable_parameter_list) > 0:
            # do update of model parameters
            optimizer.zero_grad()
            if isinstance(module_output, torch.Tensor):
                module_target = module_target.to(device=module_output.device)
                loss = loss_function(module_output, target_tensor)
            elif isinstance(module_output, dict):
                if "logits" in module_output:
                    module_target = module_target.to(device=module_output["logits"].device)
                    loss = loss_function(module_output["logits"], target_tensor)
                elif ENCODER_OUTPUT in module_output:
                    module_target = module_target.to(device=module_output[ENCODER_OUTPUT].device)
                    loss = loss_function(module_output[ENCODER_OUTPUT], target_tensor)
                elif "combiner_output" in module_output:
                    module_target = module_target.to(device=module_output["combiner_output"].device)
                    loss = loss_function(module_output["combiner_output"], target_tensor)
            elif isinstance(module_output, (list, tuple)):
                module_target = module_target.to(device=module_output[0].device)
                loss = loss_function(module_output[0], target_tensor)
            else:
                raise ValueError(f"Unexpected output type.  Module type found is {type(module_output)}")

            loss.backward()
            optimizer.step()

            # check for parameter updates
            parameter_updated = []
            # create tuple for each parameter: (parameter name, update indicator True/False)
            # parameter is deemed updated if the gradient is not None and the gradient has non-zero value
            for p in module.named_parameters():
                parameter_updated.append((p[0], (p[1].grad is not None) and (not torch.all(p[1].grad == 0))))
        else:
            parameter_updated = []

        parameters_not_updated = []
        for p in parameter_updated:
            # if not updated, record parameter name
            if not p[1]:
                parameters_not_updated.append(p[0])

    trainable_parameters = len(trainable_parameter_list)
    parameters_updated = sum(p[1] for p in parameter_updated)
    frozen_parameters = len(frozen_parameter_list)

    return frozen_parameters, trainable_parameters, parameters_updated, parameters_not_updated
