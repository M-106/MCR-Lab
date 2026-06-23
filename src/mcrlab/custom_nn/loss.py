# -----------
# > Imports <
# -----------
import torch.nn as nn



# ----------
# > Getter <
# ----------
def get_criterion(name:str, check_valid=True):
    name = name.lower()

    criterion = None

    if name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    # ...

    if check_valid:
        if criterion is None:
            raise ValueError(f"Could not create Criterion/Loss with name '{name}'.")

    return criterion








