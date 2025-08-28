import torch.nn as nn


def get_module_training_status(model: nn.Module) -> dict:
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


def set_eval_mode_frozen_modules(model: nn.Module) -> None:
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
        param_count += get_param_counts(module)["__total__"]
        ret_dict[module_name] = get_param_counts(module)

    # Add up direct parameters (not in submodules)
    for param in model.parameters(recurse=False):
        param_count += param.numel()

    ret_dict["__class__"] = str(model.__class__.__name__)
    ret_dict["__total__"] = param_count

    return ret_dict


def prettify_param_counts(param_counts: dict, max_depth: int = 2) -> dict:
    """
    Prettify the parameter counts.
    Output json like 
    {
        "module_name (class_name, num_params in ... (same))": {
            "submodule_name (class_name, num_params in Billion (when >= 1B) or Million (when >= 1M) or Kilo (else))": {
                "subsubmodule_name (class_name, num_params in ... (same as above))": {
                    ...
                }
            }
        }
    }
    """
    ret_dict = dict()
    for module_name, module_counts in param_counts.items():
        if module_name == "__total__":
            ret_dict["__total__"] = f"{module_counts / 1e9:.3f}B"
        elif module_name == "__class__":
            continue  # Skip class info in prettified output
        else:
            # Get class name and total params for this module
            class_name = module_counts.get("__class__", "Unknown")
            
            total_params = module_counts.get("__total__", 0)
            if total_params >= 1e9:
                param_str = f"{total_params / 1e9:.3f}B"
            elif total_params >= 1e6:
                param_str = f"{total_params / 1e6:.3f}M"
            else:
                param_str = f"{total_params / 1e3:.1f}K"
            
            pretty_key = f"{module_name} ({class_name}, {param_str})"
            
            # Recursively prettify submodules
            if max_depth > 0:
                ret_dict[pretty_key] = prettify_param_counts(module_counts, max_depth - 1)
    
    return ret_dict