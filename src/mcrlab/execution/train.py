# -----------
# > Imports <
# -----------
import os
import shutil
from functools import partial

from tqdm import tqdm
from transformers import (Trainer as HFTrainer, 
                         TrainingArguments as HFTrainingArguments, 
                         SegformerForSemanticSegmentation, 
                         Mask2FormerForUniversalSegmentation, 
                         OneFormerForUniversalSegmentation,
                         # DeepLabV3ForSemanticSegmentation,
                         SegformerImageProcessor,
                         AutoImageProcessor)
                         #TrainerCallBack as HFTrainerCallBack

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

# for plotting
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl

from mcrlab.config.config import Config
from mcrlab.log import get_logger, LoggerPrinter
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.model_utils import get_model, get_device, get_criterion
from mcrlab.metrices import compute_metrics


# ----------------------
# > HuggingFace Helper <
# ----------------------
class ImagePlottingCallback(TrainerCallback):
    def __init__(self, val_dataset, model_name, processor, config, num_samples=1, pre_name="", clear_path=True, save_post_dir_name=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.model_name = model_name.lower()
        self.processor = processor
        self.config = config
        self.num_samples = num_samples
        self.pre_name = pre_name
        
        # create folderfor saving
        self.plot_dir = f"./output/plots/{model_name}"
        if save_post_dir_name is not None:
            self.plot_dir += f"_{save_post_dir_name}"
        os.makedirs(self.plot_dir, exist_ok=True)
        if clear_path:
            shutil.rmtree(self.plot_dir)
            os.makedirs(self.plot_dir, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        Calls after every evaluation
        """
        if model is None:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        plotted_samples = 0

        # go through the first x samples
        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                sample = self.val_dataset[i]
                
                # Prepare inputs -> add batch dim + move to device
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                labels = sample["labels"]  # Ground Truth (bereits als Tensor im Dataset)

                # make prediction
                outputs = model(pixel_values=pixel_values)
                
                # extract target_sizes if needed (for Mask2Former/OneFormer?)
                target_sizes = [labels.shape[-2:]] if self.model_name in ["mask2former", "oneformer"] else None
                
                preds = get_segmentation_prediction(
                    outputs,
                    model_name=self.model_name,
                    processor=self.processor,
                    target_sizes=target_sizes
                )
                
                input_img = pixel_values[0].cpu().numpy().transpose(1, 2, 0)
                
                if input_img.shape[-1] == 3:
                    # if input_img.max() > 1:
                    #     print(f"[WARNING] Found value bigger than 1 ({input_img.max()}), will clip it away for visualization.")
                    # if input_img.min() < 0:
                    #     print(f"[WARNING] Found value smaller than 0 ({input_img.min()}), will clip it away for visualization.")
                    
                    input_to_show = np.clip(((input_img-input_img.min())/(input_img.max() - input_img.min())), 0, 1) 
                else:
                    input_to_show = input_img[:, :, 0]

                gt_mask = labels.cpu().numpy().squeeze()
                # torch.as_tensor(labels)
                pred_mask = preds[0].cpu().numpy().squeeze()

                if np.sum(gt_mask == 1) < 25:
                    continue

                # "remove" ignore label, so it does not hinder the plot
                gt_mask = np.where(gt_mask == 255, 0, gt_mask)

                # print(f"[DEBUG] GT unique values: {np.unique(gt_mask)} | Pred unique values: {np.unique(pred_mask)}")

                # # scale values
                # if gt_mask.max() <= 1:
                #     gt_mask *= 255

                # if pred_mask.max() <= 1:
                #     pred_mask *= 255

                # create plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(input_to_show)
                axes[0].set_title("Input (Image/BEV)")
                axes[0].axis("off")

                axes[1].imshow(gt_mask, cmap='viridis', vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=1)
                axes[2].set_title(f"Prediction (Epoch {state.epoch:.1f})")
                axes[2].axis("off")

                save_path = os.path.join(self.plot_dir, f"{self.pre_name}_epoch_{state.epoch:03}_step_{state.global_step:03}_sample_{i:03}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

                plotted_samples += 1

                if plotted_samples >= self.num_samples:
                    break
    
        if was_training:
            model.train()



# ----------
# > Helper <
# ----------
def save_model(model, dir_path, name="model"):
    os.makedirs(dir_path, exist_ok=True)

    if not name.endswith(".pt"):
        name += ".pt"

    torch.save(model.state_dict(), os.path.join(dir_path, name))

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scaler(name):
    if name:
        return torch.amp.GradScaler()
    return None

def get_scheduler(name, optimizer):
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return None



# -----------------
# > Trainer Class <
# -----------------
class Trainer:
    def __init__(self, 
                 model,
                 name,
                 criterion,
                 optimizer,
                 epochs,
                 data_loader,
                 val_data_loader,
                 val_steps=5,
                 lr_scheduler=None,
                 device=None,
                 use_amp=False,
                 scaler=False,
                 # logging_steps=10,
                 output_dir="./results",
                 checkpoint_best_model=True,
                 logger=LoggerPrinter,
                 writer=None):
        
        self.model = model
        self.name = name
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.val_steps = val_steps
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.use_amp = use_amp
        self.scaler = scaler
        # self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.checkpoint_best_model = checkpoint_best_model
        self.logger = logger
        self.writer = writer

        if self.device is None:
            self.device = get_device()

        self.model.to(self.device)



    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0  # float("inf")
        processed_batches = 0

        for batch_idx, (x_batches, y_batches) in tqdm(enumerate(self.data_loader), desc=f"Training, Epoch {epoch}"):
            # move data to device
            x_batches = x_batches.to(self.device)
            y_batches = y_batches.to(self.device)

            if self.use_amp and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    predictions = self.model(x_batches)
                    loss = self.criterion(predictions, y_batches)
            else:
                predictions = self.model(x_batches)
                loss = self.criterion(predictions, y_batches)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # some scheduler want per epoch!!

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", loss.item(), processed_batches)

            total_loss += loss.item()
            processed_batches += 1

        # mean loss
        return total_loss / max(processed_batches, 1)
    


    def evaluate(self):
        if self.val_data_loader is None:
            return 0.0  # float("inf")
        
        self.model.eval()
        total_loss = 0.0
        # other metrices maybe ...
        processed_batches = 0

        # or: no_grad -> but inference mode is quicker
        with torch.inference_mode():
            for batch_idx, (x_batches, y_batches) in tqdm(enumerate(self.val_data_loader), desc=f"Validation"):
                x_batches = x_batches.to(self.device)
                y_batches = y_batches.to(self.device)

                predictions = self.model(x_batches)
                loss = self.criterion(predictions, y_batches)

                total_loss += loss.item()
                processed_batches += 1

        return total_loss / max(processed_batches, 1)



    def train(self):
        self.best_val_loss = float("inf")

        for epoch in tqdm(range(self.epochs), desc="Epoch", total=self.epochs):
            loss = self.train_epoch(epoch)

            if epoch % self.val_steps == 0:
                val_loss = self.evaluate()

                if self.checkpoint_best_model and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_model(self.model, self.output_dir, name="best_"+self.name)

            # experiment tracking
            # ...

            # checkpoint saving
            if not self.checkpoint_best_model and epoch % 5 == 0:
                save_model(self.model, self.output_dir, name=self.name)

        self.logger.info("Congratulations the training is finish.")
        # add more loggings ...



# # -----------------------
# # > Main Train Pipeline <
# # -----------------------
# def train_pipeline(config):
#     # extract config settings
#     checkpoint_dir = os.path.join(config.train.checkpoint_dir, config.model.name)
#     batch_size = config.train.batch_size
#     learning_rate = config.train.learning_rate
#     use_amp = config.train.use_amp
#     scaler_name = config.train.scaler
#     criterion_name = config.train.criterion
#     optimizer_name = config.train.optimizer
#     best_model = config.train.checkpoint_best_model
#     val_steps = config.train.val_steps
#     lr_scheduler = config.train.lr_scheduler

#     model_name = config.model.name

#     # load model
#     model = get_model(config.model.name)
#     # model = MLP()
#     # model.load_state_dict(torch.load(""))

#     # load data
#     data_loader = get_data_loader(config.data.name, config.data.path, 
#                                     testdata=False, 
#                                     transform=get_basic_transform(num_points=-1), # get_basic_transform(num_points=-1),
#                                     batch_size=batch_size, shuffle=True, num_workers=1,
#                                     preprocessed=True,
#                                     return_train_format=True)
#     dataset = data_loader.dataset

#     train_size = int(0.8*len(dataset))
#     val_size = len(dataset) - train_size
#     generator = torch.Generator().manual_seed(42)

#     train_dataset, val_dataset = random_split(dataset, 
#                                               [train_size, val_size], 
#                                               generator=generator)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Experiment Tracking
#     writer = SummaryWriter()

#     optimizer = get_optimizer(optimizer_name, model, learning_rate)

#     # start training
#     trainer = Trainer(
#         model=model,
#         name=model_name,
#         criterion=get_criterion(criterion_name),
#         optimizer=optimizer,
#         epochs=config.train.epochs,
#         data_loader=train_loader,
#         val_data_loader=val_loader,
#         val_steps=val_steps,
#         lr_scheduler=get_scheduler(lr_scheduler, optimizer),
#         device=get_device(config.device),
#         use_amp=use_amp,
#         scaler=get_scaler(scaler_name),
#         output_dir=checkpoint_dir,
#         checkpoint_best_model=best_model,
#         writer=writer
#     )

#     trainer.train()

#     writer.close()



# ------------------------------
# > HuggingFace Train Pipeline <
# ------------------------------
# FIXME: Checkpoint loading making more generell or all like with segformer??
# def get_model_and_preprocessor(model_name, check_point_path=None):
#     if model_name == "segformer":
#         if check_point_path is not None:
#             model = SegformerForSemanticSegmentation.from_pretrained(
#                 check_point_path,  # just path to the folder!
#                 num_labels=2,
#                 ignore_mismatched_sizes=True
#             )
#             # or
#             # model = SegformerForSemanticSegmentation.from_pretrained(model_name)
#             # state_dict = torch.load("pytorch_model.bin", map_location="cpu")
#             # model.load_state_dict(state_dict)
#         else:
#             model = SegformerForSemanticSegmentation.from_pretrained(
#                 "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
#                 num_labels=2,
#                 ignore_mismatched_sizes=True
#             )

#         model.config.ignore_index = 255  # or -1?
#         model.config.num_labels = 2
#         # model.config.id2label = {...}
#         # model.config.label2id = {...}

#         preprocessor = SegformerImageProcessor.from_pretrained(
#             "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
#             do_resize=True,
#             size={"height": 512, "width": 512},   # match your tile size
#             do_rescale=False,    # ← important: your data is already float
#             do_normalize=True,
#             image_mean=[0.485, 0.456, 0.406],     # ImageNet stats, or use
#             image_std=[0.229, 0.224, 0.225],      # your own dataset stats
#         )
#     elif model_name == "mask2former":
#         model = Mask2FormerForUniversalSegmentation.from_pretrained(
#             "facebook/mask2former-swin-large-cityscapes-semantic",
#             num_labels=2,
#             ignore_mismatched_sizes=True
#         )
#         model.config.ignore_index = 255
#         model.config.num_labels = 2

#         preprocessor = AutoImageProcessor.from_pretrained(
#             "facebook/mask2former-swin-large-cityscapes-semantic",
#             do_resize=True,
#             size={"shortest_edge": 512},
#             do_rescale=False,    # ← same: already float tensor
#             do_normalize=True,
#             image_mean=[0.485, 0.456, 0.406],
#             image_std=[0.229, 0.224, 0.225],
#         )
#     elif model_name == "oneformer":
#         model = OneFormerForUniversalSegmentation.from_pretrained(
#             "shi-labs/oneformer_cityscapes_swin-l_160k",
#             num_labels=2,
#             ignore_mismatched_sizes=True
#         )
#         model.config.ignore_index = 255
#         model.config.num_labels = 2

#         preprocessor = AutoImageProcessor.from_pretrained(
#             "shi-labs/oneformer_cityscapes_swin-l_160k",
#             do_resize=True,
#             size={"shortest_edge": 512},
#             do_rescale=False,
#             do_normalize=True,
#             image_mean=[0.485, 0.456, 0.406],
#             image_std=[0.229, 0.224, 0.225],
#         )
#     elif model_name == "deeplabv3":
#         model = DeepLabV3ForSemanticSegmentation.from_pretrained(
#             "microsoft/deeplabv3-resnet-101",
#             num_labels=2,
#             ignore_mismatched_sizes=True
#         )
#         model.config.ignore_index = 255
#         model.config.num_labels = 2
    
#         preprocessor = AutoImageProcessor.from_pretrained(
#             "microsoft/deeplabv3-resnet-101",  # or resnet-50
#             do_resize=True,
#             size={"height": 512, "width": 512},
#             do_rescale=False,      # already float32
#             do_normalize=True,
#             image_mean=[0.485, 0.456, 0.406],
#             image_std=[0.229, 0.224, 0.225],
#         )
#     else:
#         raise ValueError(f"Does not support model '{model_name}'")
    
#     return model, preprocessor
# Model registry: name -> (model_class, default_checkpoint)
MODEL_REGISTRY = {
    "segformer":   (SegformerForSemanticSegmentation,        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"),
    "mask2former": (Mask2FormerForUniversalSegmentation,     "facebook/mask2former-swin-large-cityscapes-semantic"),
    "oneformer":   (OneFormerForUniversalSegmentation,       "shi-labs/oneformer_cityscapes_swin-l_160k"),
    # "deeplabv3":   (DeepLabV3ForSemanticSegmentation,        "microsoft/deeplabv3-resnet-101"),
    # -> model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
}

# processor size config per model
PROCESSOR_SIZE = {
    "segformer":   {"height": 512, "width": 512},
    "mask2former": {"shortest_edge": 512},
    "oneformer":   {"shortest_edge": 512},
    # "deeplabv3":   {"height": 512, "width": 512},
}



def get_model_and_processor(model_name, check_point_path=None, num_labels=2,
                                image_mean=[0.485, 0.456, 0.406],
                                image_std=[0.229, 0.224, 0.225]):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    # MODEL EXTRACTION
    # ------------
    # extract model class and checkpoitn/default loading
    model_class, default_checkpoint = MODEL_REGISTRY[model_name]
    # checkpoint = check_point_path or default_checkpoint
    checkpoint = check_point_path if check_point_path is not None else default_checkpoint

    # MODEL LOADING
    # ------------
    model = model_class.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.config.ignore_index = 255
    model.config.num_labels = num_labels

    # PROCESSOR
    # ------------
    processor_source = default_checkpoint if check_point_path else checkpoint
    processor = AutoImageProcessor.from_pretrained(
        processor_source,
        do_resize=True,
        size=PROCESSOR_SIZE[model_name],
        do_rescale=False,
        do_normalize=True,
        image_mean=image_mean,
        image_std=image_std,
    )

    return model, processor



def get_segmentation_prediction(outputs, model_name, processor=None, target_sizes=None):
    model_name = model_name.lower()

    is_numpy_input = isinstance(outputs, np.ndarray)

    if model_name in ["segformer", "deeplabv3"]:
        if not is_numpy_input and hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        # raise RuntimeError(f"DEBUGGING STOP, logits/pred shape: {logits.shape}\nMin-Max ({logits.min()} - {logits.max()})")
        # if isinstance(logits, np.ndarray):
        #     preds = logits.argmax(axis=1)
        # else:
        #     preds = logits.argmax(dim=1)

        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits)
        else:
            logits_tensor = logits.detach().cpu()

        preds = logits_tensor.argmax(dim=1)

        # upscaling because: SegFormer logits are 1/4 of input size
        if target_sizes is not None:
            size = tuple(int(x) for x in target_sizes[0])
    
            preds = preds.unsqueeze(1).float()  # Interpolate braucht 4D: (B, 1, H, W)
            preds = F.interpolate(preds, size=size, mode="nearest")
            preds = preds.squeeze(1).long()

    elif model_name in ["mask2former", "oneformer"]:
        if is_numpy_input:
            logits_tensor = torch.from_numpy(outputs)
            preds = logits_tensor.argmax(dim=1)
            
            if target_sizes is not None:
                size = tuple(int(x) for x in target_sizes[0])
                preds = preds.unsqueeze(1).float()
                preds = F.interpolate(preds, size=size, mode="nearest")
                preds = preds.squeeze(1).long()
        else:
            if processor is None:
                raise ValueError(f"Processor muss für {model_name} übergeben werden!")
            
            # Post-processing liefert eine Liste von PyTorch-Tensoren
            preds_list = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes
            )
            preds = torch.stack(preds_list)
    else:
        raise ValueError()


    return preds.numpy() if is_numpy_input else preds


def train_hf_pipeline(config):
    print("GPU available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("Does not find GPU accelerator!")

    batch_size = 12

    model_name = config.model.name.lower()
    checkpoint_path = config.model.check_point_path
    if checkpoint_path == "None":
        checkpoint_path = None
    model, processor = get_model_and_processor(model_name, checkpoint_path)

    # Load Data
    train_dataset = get_data_loader(config.data.name, 
                                   config.data.path, 
                                   type="train", 
                                   transform=get_basic_transform(),
                                   batch_size=config.train.batch_size, 
                                   shuffle=True, 
                                   num_workers=4,
                                   preprocessed=True, 
                                   return_train_format=True,
                                   return_dataset=True)
    all_train_paths = train_dataset.point_cloud_paths
    train_dataset = BEVDataset(path=all_train_paths, 
                               file_paths=[], 
                               has_labels=True, 
                               image_training=True, 
                               preprocessor=processor,
                               augment=True)
    train_dataset.manhole_filter(required_manhole_points=50)
    
    val_dataset = get_data_loader(config.data.name, 
                                   config.data.path, 
                                   type="val", 
                                   transform=get_basic_transform(),
                                   batch_size=config.train.batch_size, 
                                   shuffle=False, 
                                   num_workers=4,
                                   preprocessed=True, 
                                   return_train_format=True,
                                   return_dataset=True)
    all_val_paths = val_dataset.point_cloud_paths
    val_dataset = BEVDataset(path=all_val_paths, 
                             file_paths=[], 
                             has_labels=True, 
                             image_training=True, 
                             preprocessor=processor,
                             augment=False)
    val_dataset.manhole_filter(required_manhole_points=50)
    # config.data.preprocessed

    # Callback for in-between sample plotting
    plotting_callback = ImagePlottingCallback(
        val_dataset=val_dataset,
        model_name=model_name,
        processor=processor,
        config=config,
        num_samples=5,
        pre_name="finetuning",
    )

    # Helper Functions
    def collate_fn(batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics_fn(eval_pred, model_name, processor):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            outputs = eval_pred.predictions
            labels = eval_pred.label_ids

            target_sizes = [label.shape[-2:] for label in labels]
        else:
            outputs, labels = eval_pred

            target_sizes = None



        preds = get_segmentation_prediction(
            outputs,
            model_name=model_name,
            processor=processor,
            target_sizes=target_sizes
        )

        return compute_metrics(preds=preds, labels=labels)

    # FIXME -> make many of the settings adjustable via config
    training_args = HFTrainingArguments(
        output_dir=f"./output/checkpoints/{config.model.name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=6e-5,           # 6e-5     
        lr_scheduler_type="cosine",   # cosine
        warmup_steps=200,             # 0.1
        fp16=False,                    # faster training
        gradient_accumulation_steps=4,
        num_train_epochs=200,   
        dataloader_num_workers=4,
        eval_strategy="steps",
        eval_steps=int( len(train_dataset)/batch_size ),
        save_strategy="steps",
        save_steps=int( len(train_dataset)/batch_size ),
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,   # important for SAM
        push_to_hub=False,
        report_to=["tensorboard", "mlflow"],  # "none"
        use_cpu=False
    )

    # for debugging
    # training_args = HFTrainingArguments(
    #     output_dir="./tmp_test",
    #     max_steps=1,
    #     logging_steps=1,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     remove_unused_columns=False,
    #     report_to="none",
    #     use_cpu=True
    # )

    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # load bev/meta files and extract in right format -> use BEV Dataset
        eval_dataset=val_dataset,
        # train_dataset=train_dataset.select(range(2)) if hasattr(train_dataset, 'select') else train_dataset,
        # eval_dataset=val_dataset.select(range(2)) if hasattr(val_dataset, 'select') else val_dataset,
        data_collator=collate_fn,
        compute_metrics=partial(
            compute_metrics_fn,
            model_name=model_name,
            processor=processor
        ),
        callbacks=[plotting_callback]
    )

    trainer.train()
    print("Success! Your Training is finish and your pipeline works.")

    # ------------------------------------
    # --> POST TRAINING <--

    # robustness finetuning
    # danger: Catastrophic Forgetting

    print("Start post training.")

    train_dataset = BEVDataset(path=all_train_paths, 
                               file_paths=[], 
                               has_labels=True, 
                               image_training=True, 
                               preprocessor=processor,
                               augment=True)
    val_dataset = BEVDataset(path=all_val_paths, 
                             file_paths=[], 
                             has_labels=True, 
                             image_training=True, 
                             preprocessor=processor,
                             augment=False)

    plotting_callback = ImagePlottingCallback(
        val_dataset=val_dataset,
        model_name=model_name,
        processor=processor,
        config=config,
        num_samples=5,
        pre_name="post_training", 
        clear_path=True,
        save_post_dir_name="post_training"
    )

    training_args = HFTrainingArguments(
        output_dir=f"./output/checkpoints/{config.model.name}_post_training",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-6,           # 6e-5     
        lr_scheduler_type="cosine",   # cosine
        warmup_steps=0,             # 0.1
        fp16=False,                    # faster training
        gradient_accumulation_steps=4,
        num_train_epochs=100,   
        dataloader_num_workers=4,
        eval_strategy="steps",
        eval_steps=int( len(train_dataset)/batch_size ) * 2,
        save_strategy="steps",  # or best?
        save_steps=int( len(train_dataset)/batch_size ) * 2,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,   # important for SAM
        push_to_hub=False,
        report_to=["tensorboard", "mlflow"],  # "none"
        use_cpu=False
    )

    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # load bev/meta files and extract in right format -> use BEV Dataset
        eval_dataset=val_dataset,
        # train_dataset=train_dataset.select(range(2)) if hasattr(train_dataset, 'select') else train_dataset,
        # eval_dataset=val_dataset.select(range(2)) if hasattr(val_dataset, 'select') else val_dataset,
        data_collator=collate_fn,
        compute_metrics=partial(
            compute_metrics_fn,
            model_name=model_name,
            processor=processor
        ),
        callbacks=[plotting_callback]
    )

    trainer.train()

    print("Post Training is finish!")



# -----------------------
# > Main Train Function <
# -----------------------
def train(config):
    # model_name = config.model.name

    train_hf_pipeline(config)

    # if model_name.lower() in ["sam2", "sam3", "segformer", "dinomask2former"]:
    #     train_hf_pipeline(config)
    # else:
    #     raise RuntimeError("Only Huggingface Pipeline is available right now")
    #     train_pipeline(config)











