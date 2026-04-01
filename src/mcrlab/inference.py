# -----------
# > Imports <
# -----------
import torch
from tqdm import tqdm

from mcrlab.config.config import Config
from mcrlab.model_utils import get_model, get_device, TorchModelWrapper
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform



# ----------------
# > Inference <
# ----------------
class Inferencer:
    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = get_device(device)

    def run(self):
        results = []

        for batch_idx, x_batch in tqdm(enumerate(self.dataloader),
                                       desc="Inference",
                                       total=len(self.dataloader)
                                    ):
            # dataloader might return (x, y)
            if isinstance(x_batch, (list, tuple)):
                x_batch = x_batch[0]

            preds = self.model.predict(x_batch)

            results.append(preds)

        return results



# ----------------
# > Main
# ----------------
def inference(config):

    device = get_device(config.device)

    # load model
    model = get_model(config.model.name)

    if isinstance(model, torch.nn.Module):
        model = TorchModelWrapper(model, device)

    # load data
    data_loader = get_data_loader(
        config.data.name,
        config.data.path,
        testdata=True,   
        transform=get_basic_transform(num_points=-1),
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=1,
        preprocessed=True,
        return_train_format=False  # only x
    )

    inferencer = Inferencer(
        model=model,
        dataloader=data_loader,
        device=device
    )

    results = inferencer.run()

   # for debugging
    for i, res in enumerate(results[:5]):
        print(f"Sample {i}: {res}")

    return results











