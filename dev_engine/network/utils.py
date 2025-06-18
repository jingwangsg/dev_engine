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

def get_param_counts(model: nn.Module):
    """
    Get the number of parameters recursively.
    """
    param_count = 0
    ret_dict = dict()
    
    # Count parameters in submodules first
    for module_name, module in model.named_children():
        param_count += get_param_counts(module)
        ret_dict[module_name] = get_param_counts(module)
    
    # Add up direct parameters (not in submodules)
    for param in model.parameters(recurse=False):
        param_count += param.numel()
    
    ret_dict["__class__"] = model.__class__
    ret_dict["__total__"] = param_count
    
    return ret_dict
