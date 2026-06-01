# -----------
# > Imports <
# -----------
import os
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
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from mcrlab.config.config import Config
from mcrlab.log import get_logger, LoggerPrinter
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.model_utils import get_model, get_device, get_criterion
from mcrlab.metrices import compute_metrics



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

    if model_name in ["segformer", "deeplabv3"]:
        logits = outputs.logits
        return logits.argmax(dim=1)

    elif model_name in ["mask2former", "oneformer"]:
        preds = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes
        )

        return torch.stack(preds)
    else:
        raise ValueError()



def train_hf_pipeline(config):
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
    # config.data.preprocessed

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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=6e-5,           # 6e-5     
        lr_scheduler_type="cosine",   # cosine
        warmup_ratio=0.1,             # 0.1
        fp16=False,                    # faster training
        gradient_accumulation_steps=4,
        num_train_epochs=50,   
        dataloader_num_workers=4,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,   # important for SAM
        push_to_hub=False,
        report_to=["tensorboard", "mlflow"]  # "none"
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
        )
    )

    trainer.train()
    print("Success! The pipeline works.")



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











