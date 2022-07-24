import logging
from typing import Callable, Dict, Tuple, Union

import torch

from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)


class ParameterUpdateError(Exception):
    pass


# TODO: do we need this version of parameter update checking
def assert_module_parameters_updated(
    module: LudwigModule,
    module_input_args: Tuple,
    max_steps: int = 1,
    threshold: float = 1.0,
    learning_rate: float = 0.001,
) -> None:
    """
    Confirms that module parameters can be updated.
    Args:
        module: (LudwigModel) model to be tested.
        module_input_args: (tuple) input for model
        max_steps: (int, default=1) maximum number of steps allowed to test for parameter
            updates.
        threshold: (float, default=1.0) fraction of parameters that need to be updated
            to pass this test.
        learning_rate: (flaot, default=0.001) learning rate for the optimizaer

    Returns: None

    """
    # setup
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    module.train(True)

    # generate initial model output tensor
    module_output = module(*module_input_args)

    # create target tensor
    if isinstance(module_output, torch.Tensor):
        target_tensor = torch.randn(module_output.shape, dtype=module_output.dtype)
    elif isinstance(module_output, tuple):
        target_tensor = torch.randn(module_output[0].shape, dtype=module_output[0].dtype)
    else:
        raise RuntimeError("Unable to setup target tensor for model parameter update testing.")

    for step in range(max_steps):
        # make pass through model
        model_output = module(*module_input_args)

        # capture model parameters before doing parameter update pass
        before = [(x[0], x[1].clone()) for x in module.named_parameters()]

        # do update of model parameters
        optimizer.zero_grad()
        if isinstance(model_output, torch.Tensor):
            loss = loss_function(model_output, target_tensor)
        else:
            loss = loss_function(model_output[0], target_tensor)
        loss.backward()
        optimizer.step()

        # capture model parameters after a pass
        after = [(x[0], x[1].clone()) for x in module.named_parameters()]

        # check for parameter updates
        parameter_updated = []
        for b, a in zip(before, after):
            parameter_updated.append((a[1] != b[1]).any())

        # if parameters were updated in all layers, the exit loop
        if all(parameter_updated):
            logger.debug(f"\nall model parameters updated at step {step + 1}")
            # early stop
            break

    # if not all layers are updated, raise exception
    parameter_fraction_updated = float(sum(parameter_updated)) / len(parameter_updated)
    # TODO: turn print() to logger.debug() call before final merge
    print(
        f"number parameters: {len(parameter_updated)}, number updated: {sum(parameter_updated)}"
        f", fraction: {parameter_fraction_updated:0.2f}"
    )
    if not (all(parameter_updated) or (parameter_fraction_updated >= threshold)):
        parameters_not_updated = []
        for updated, b, a in zip(parameter_updated, before, after):
            if not updated:
                parameters_not_updated.append(
                    f"\n\tParameter {a[0]} not updated.\n"
                    f"\tbefore values (requires grad:{b[1].requires_grad}): {b[1]} {b[1].grad}\n"
                    f"\tafter values (requires grad:{a[1].requires_grad}): {a[1]} {a[1].grad}\n"
                )
        raise ParameterUpdateError(
            f"Not all model parameters updated after {max_steps} iteration(s):" f"{''.join(parameters_not_updated)}"
        )


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
                loss = loss_function(module_output, target_tensor)
            elif isinstance(module_output, dict):
                if "logits" in module_output:
                    loss = loss_function(module_output["logits"], target_tensor)
                elif "encoder_output" in module_output:
                    loss = loss_function(module_output["encoder_output"], target_tensor)
            else:
                loss = loss_function(module_output[0], target_tensor)

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


def _assert_module_parameters_updated(
    module: LudwigModule,
    module_input: torch.Tensor,
    extract_module_output: Callable[[Union[Dict, Tuple]], torch.Tensor],
    max_steps: int = 1,
) -> None:
    """
    Confirms that model parameters can be updated.
    Args:
        module: (LudwigModel) moddule to be tested.
        module_input: (torch.Tensor) input for model
        extract_module_output: (Callable[[Union[Dict, Tuple]], torch.Tensor])
            function to extract tensor for use in parameter update testing
        max_steps: (int) maximum number of steps allowed to test for parameter
            updates.

    Returns: None

    """
    # setup
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.001)
    module.train(True)

    # create target tensor
    module_output = extract_module_output(module(module_input))
    target_tensor = torch.randn(module_output.shape, dtype=module_output.dtype)

    step = 1
    while True:
        # capture model parameters before doing parameter update pass
        before = [(x[0], x[1].clone()) for x in module.named_parameters()]

        # do update of model parameters
        optimizer.zero_grad()
        loss = loss_function(module_output, target_tensor)
        loss.backward()
        optimizer.step()

        # capture model parameters after one pass
        after = [(x[0], x[1].clone()) for x in module.named_parameters()]

        # check for parameter updates
        parameter_updated = []
        for b, a in zip(before, after):
            parameter_updated.append((a[1] != b[1]).any())

        # check to see if parameters were updated in all layers
        if all(parameter_updated):
            logger.debug(f"\nall model parameters updated at step {step}")
            break
        elif step >= max_steps:
            # exceeded maximum allowed tries, terminating with error
            parameters_not_updated = []
            for updated, b, a in zip(parameter_updated, before, after):
                if not updated:
                    parameters_not_updated.append(
                        f"\n\tParameter {a[0]} not updated."
                        # f"\tbefore model forward() pass (requires grad:{b[1].requires_grad}): {b[1]}\n"
                        # f"\tafter model forward() pass (requires grad:{a[1].requires_grad}): {a[1]}\n"
                    )
            raise ParameterUpdateError(
                f"Not all model parameters updated after {step} iteration(s):" f"{''.join(parameters_not_updated)}"
            )

        # make another pass through model
        module_output = extract_module_output(module(module_input))
        step += 1


def assert_model_parameters_updated_encoders(
    module: LudwigModule, module_input: torch.Tensor, max_steps: int = 1
) -> None:
    """
    Confirms that model parameters can be updated.
    Args:
        model: (Type[Encoder]) model to be tested.
        model_input: (torch.Tensor) input for model
        max_steps: (int) maximum number of steps allowed to test for parameter
            updates.

    Returns: None

    """
    _assert_module_parameters_updated(module, module_input, lambda outputs: outputs["encoder_output"], max_steps)
