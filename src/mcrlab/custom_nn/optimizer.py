# -----------
# > Imports <
# -----------
import torch



# ----------
# > Getter <
# ----------
def get_optimizer(name:str, model:torch.nn.Module, lr:float, check_valid=True):
    name = name.lower()

    optimizer = None

    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters, lr=lr)
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # ...

    if check_valid:
        if optimizer is None:
            raise ValueError(f"Could not create Optimizer with name '{name}'.")

    return optimizer







