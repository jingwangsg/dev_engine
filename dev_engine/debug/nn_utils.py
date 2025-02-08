def param_info(model):
    param_info = {}

    # Iterate over all named parameters in the model.
    for name, param in model.named_parameters():
        # Create the info dictionary for this parameter.
        info = {
            "shape": list(param.shape),
            "requires_grad": param.requires_grad,
            "grad_norm": param.grad.norm().item() if param.grad is not None else None,
        }

        # Split the parameter name by dots to create a nested dictionary.
        keys = name.split(".")
        current = param_info
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        # Set the final key to the info dict.
        current[keys[-1]] = info

    return param_info
