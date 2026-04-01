# -----------
# > Imports <
# -----------
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mcrlab.config.config import Config
from mcrlab.model_utils import get_model, get_device, get_criterion, \
                               TorchModelWrapper, \
                               compute_loss, match_with_thresholding, \
                               compute_metrics
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform
        


# -----------
# > Testing <
# -----------
class Tester:
    def __init__(self, model, dataloader, loss_fns, device=None):
        self.model = model
        self.dataloader = dataloader
        self.loss_fns = loss_fns
        self.device = get_device(device)

    def evaluate(self):
        self.model.eval()

        total_losses = [0.0 for _ in self.loss_fns]
        # maybe other metrices?

        batches = len(self.dataloader)

        # with torch.no_grad():
        with torch.inference_mode():
            for batch_idx, (x_batch, y_batch) in tqdm(enumerate(self.dataloader), desc="Testing", total=batches):
                # moving data to device is not done
                # because of geometry models
                predictions = self.model.predict(x_batch)

                matches, unmatched_pred, unmatched_gt = match_with_thresholding(pred=predictions,
                                                                                target=y_batch,
                                                                                max_dist=0.5)

                # calc loss 
                for loss_idx, loss_fn in enumerate(self.loss_fns):
                    if isinstance(loss_fn, str) and loss_fn in ["recall", "precision", "f1", "f2"]:
                        cur_loss = compute_metrics(matches, unmatched_pred, unmatched_gt)[loss_fn]
                    else:
                        cur_loss = compute_loss(loss_fn=loss_fn,
                                                pred=predictions,
                                                target=y_batch,
                                                matches=matches,
                                                unmatched_gt=unmatched_gt,
                                                unmatched_pred=unmatched_pred,
                                                lambda_fn=1.0,
                                                lambda_fp=0.5)
                    total_losses[loss_idx] += cur_loss.item()

        # calc mean
        mean_losses = [cur_loss / max(batches, 1) for cur_loss in total_losses]
        return mean_losses



def test(config):

    # load model
    model = get_model(config.model.name)

    if isinstance(model, torch.nn.Module):
        model = TorchModelWrapper(model, config.device)

    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=True, 
                                    transform=get_basic_transform(num_points=-1), # get_basic_transform(num_points=-1),
                                    batch_size=config.test.batch_size, shuffle=False, num_workers=1,
                                    preprocessed=True,
                                    return_train_format=True)  # because we still want x and y
    
    metrices = []
    for metric_name in config.test.metrices:
        metrices.append(get_criterion(metric_name))

    tester = Tester(model=model, 
                    device=config.device,
                    dataloader=data_loader,
                    loss_fns=metrices)
    tester.evaluate()











