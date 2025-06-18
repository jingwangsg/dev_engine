import torch
import torch.nn as nn


def get_module_training_status(model: nn.Module):
    """
    return the training status of each module in the model recursively.
    True: all parameters are trainable
    False: all parameters are frozen
    Return:
        dict: {module_name: training_status, module_name: {submodule_name: training_status, ...}}
    """
    training_status = dict()
    for module_name, module in model.named_children():
        key = f"{module_name}({module.__class__.__name__})"
        requires_grad_status = [param.requires_grad for param in module.parameters()]

        if all([requires_grad for requires_grad in requires_grad_status]):
            training_status[key] = True
        elif all([not requires_grad for requires_grad in requires_grad_status]):
            training_status[key] = False
        else:
            training_status[key] = get_module_training_status(module)

    return training_status


def set_eval_mode_frozen_modules(model: nn.Module):
    """
    Set module recursively to evaluation mode if all parameters are frozen.
    """
    for module_name, module in model.named_children():
        requires_grad_status = [param.requires_grad for param in module.parameters()]
        if all([not requires_grad for requires_grad in requires_grad_status]):
            module.eval()
        elif all([requires_grad for requires_grad in requires_grad_status]):
            module.train()
        else:
            set_eval_mode_frozen_modules(module)
