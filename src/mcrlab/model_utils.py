# -----------
# > Imports <
# -----------
import torch
from scipy.optimize import linear_sum_assignment  # hungarian algorithm
# from pytorch3d.loss import chamfer_distance

from mcrlab.models.segmentation import SAM2, SAM3, SegFormer, \
                                       DinoMask2Former


# ---------
# > Utils <
# ---------
def get_model(name):
    name_ = name.lower()

    if name_ == "sam2":
        return SAM2()
    elif name_ == "sam3":
        return SAM3
    elif name_ == "segformer":
        return SegFormer()
    elif name_ == "dinomask2former":
        return DinoMask2Former()
    else:
        raise ValueError(f"Error during model loading. Can't find model with name '{name}'.")



def get_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)
    


# ----------
# > Helper <
# ----------
# PyTorch Wrapper
class TorchModelWrapper:
    """
    Because of the other methods might work on CPU,
    we have to do device stuff inside the wrapper 
    and not in the test-loop as usual.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.eval()

    def predict(self, x):
        x = x.to(self.device)
        # self.model.to(self.device)  # ?

        with torch.inference_mode():
            return self.model(x).cpu()

    
    
# --------
# > Loss <
# --------

# Loss Formats:
# # Ground Truth
# GT = [(x1,y1,z1), (x2,y2,z2), ...]

# # Predictions
# PRED = [(x1,y1,z1), (x2,y2,z2), ...]

# Loss need matching of points and "not found"-handling
# -> from scipy.optimize import linear_sum_assignment
# with
def compute_cost_matrix(pred, target):
    # pred: (N, 3)
    # target: (M, 3)
    return torch.cdist(pred, target)  # pairwise distances

def match(pred, target):
    cost = compute_cost_matrix(pred, target).cpu().numpy()

    row_idxs, col_idxs = linear_sum_assignment(cost)

    return cost, row_idxs, col_idxs  # indices of matches

def match_with_thresholding(pred, target, max_dist=0.5):
    matches = []
    unmatched_pred = set(range(len(pred)))
    unmatched_gt = set(range(len(target)))

    cost, row_ind, col_ind = match(pred, target)

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matches.append((r, c))
            unmatched_pred.discard(r)
            unmatched_gt.discard(c)

    return matches, unmatched_pred, unmatched_gt

def compute_metrics(matches, unmatched_pred, unmatched_gt):
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    # accuracy
    precision = tp / (tp + fp + 1e-6)

    # coverage
    recall = tp / (tp + fn + 1e-6)

    # balanced recall and precision
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    # balanced but recall weighted higher
    beta2 = 2**2
    f2_score = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-6)

    return {"precision": precision, 
            "recall": recall, 
            "f1": f1_score, 
            "f2": f2_score}

def compute_loss(loss_fn, pred, target, 
                 matches, unmatched_pred, unmatched_gt,
                 lambda_fn=1.0, lambda_fp=0.5):
    loss = 0.0

    for pred_idx, target_idx in matches:
        loss += loss_fn(pred[pred_idx],
                        target[target_idx])
        
    # handle unmatched targets -> missed detection
    # false negatives
    loss += len(unmatched_gt) * lambda_fn

    # handle unmatched predictions -> false positive
    loss += len(unmatched_pred) * lambda_fp

    # normalize loss
    num = max(len(matches), 1)
    loss = loss / num

    return loss

# Center Losses
def l1_loss(pred, target):  # distance loss
    return torch.norm(pred - target)

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    return torch.where(diff < beta,
                       0.5 * diff**2 / beta,
                       diff - 0.5*beta).mean()

def l2_loss(pred, target):
    return torch.norm(pred - target, p=2)

def weighted_l2_loss(pred, target, weights=(2.0, 2.0, 1.0)):
    w = torch.tensor(weights, device=pred.device)
    return torch.norm((pred - target)*w)

def mse_loss(pred, target):
    return torch.mean((pred - target)**2)

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt(((pred - target) ** 2).sum() + eps)

# BBox Losses
# FIXME -> iou, bbox loss with iou, dice loss, mask loss...



def get_criterion(loss_name):
    if loss_name == "l1":
        return l1_loss
    elif loss_name == "l2":
        return l2_loss
    elif loss_name == "mse":
        return mse_loss
    elif loss_name == "smooth_l1":
        return smooth_l1_loss
    elif loss_name == "charbonnier":
        return charbonnier_loss
    
    # special
    elif loss_name in ["recall", "precision", "f1", "f2"]:
        return loss_name  # must be processed somewhere else, because of different input values
    else:
        raise ValueError(f"Unknown loss: {loss_name}")






