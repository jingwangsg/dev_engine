import pandas as pd

def param_info(model, file="example.csv"):
    param_info = {}
    df = []
    # Iterate over all named parameters in the model.
    for name, param in model.named_parameters():
        # Create the info dictionary for this parameter.
        info = {
            "name": name,
            "shape": list(param.shape),
            "requires_grad": param.requires_grad,
            "grad_norm": param.grad.norm().item() if param.grad is not None else None,
            "dtype": str(param.dtype),
            "grad.dtype": str(param.grad.dtype) if param.grad is not None else None,
        }
        df += [info]
    df = pd.DataFrame(df)
    df = df.set_index("name")
    df = df.sort_index()
    df.to_csv(file, index=True)
