# -----------
# > Imports <
# -----------
from datetime import datetime

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from mcrlab.custom_nn.data import get_data
from mcrlab.custom_nn.loss import get_criterion
from mcrlab.custom_nn.optimizer import get_optimizer
from mcrlab.custom_nn.model import get_model
from mcrlab.custom_nn.metrics import get_metrics_aggregator
from mcrlab.custom_nn.scheduler import get_scheduler
from mcrlab.custom_nn.plot import plot_2d_training_samples



# ------------------
# > Train Pipeline <
# ------------------
def train_pipeline(epochs, batch_size,
                   data_name, data_path, 
                   experiment_name,
                   training_in_2d=True, 
                   sample_required_manhole_points=50, 
                   amount_non_manhole_samples=10,
                   model_check_point_path=None,
                   model_name="unet",
                   optimizer_name="adam",
                   learning_rate=3e-4,
                   criterion_name="bce",
                   scheduler_name="step",
                   metrics_aggregator_name="simple_loss",
                   config_for_logging=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.accelerator.current_accelerator()

    # load data
    train_data = get_data(
        name=data_name, 
        path=data_path, 
        type="train", 
        load_2d=training_in_2d,
        pass_label_in_preprocessor=False,
        sample_required_manhole_points=sample_required_manhole_points,
        amount_non_manhole_samples=amount_non_manhole_samples,
        return_as_dataloader=True,
        dataloader_batchsize=batch_size,
        dataloader_shuffle=True,
        dataloader_num_workers=4,
        dataloader_collate_fn=None    # for now this should be fine
    )
    val_data = get_data(
        name=data_name, 
        path=data_path, 
        type="val", 
        load_2d=training_in_2d,
        pass_label_in_preprocessor=False,
        sample_required_manhole_points=sample_required_manhole_points,
        amount_non_manhole_samples=amount_non_manhole_samples,
        return_as_dataloader=True,
        dataloader_batchsize=batch_size,
        dataloader_shuffle=False,
        dataloader_num_workers=4,
        dataloader_collate_fn=None    # for now this should be fine
    )

    # load model
    model = get_model(
        name=model_name, 
        check_point_path=model_check_point_path, 
        device=None, 
        check_valid=True
    )

    # load optimizer
    optimizer = get_optimizer(name=optimizer_name, model=model, lr=learning_rate, check_valid=True)

    # load criterion/loss
    criterion = get_criterion(name=criterion_name, check_valid=True)

    # load scheduler
    scheduler = get_scheduler(
        name=scheduler_name, 
        optimizer=optimizer, 
        warmup_epochs=0,
        check_valid=True,
        step_size=7, 
        gamma=0.1,
        **kwargs
    )

    # to device -> data of course then on the fly, streaming approach
    model = model.to(device)
    criterion = criterion.to(device)

    # metrics aggregator
        # also any additional named kwargs are possible as paramter
    train_metrics_aggregator = get_metrics_aggregator(name=metrics_aggregator_name, prename="train", check_valid=True)
    val_metrics_aggregator = get_metrics_aggregator(name=metrics_aggregator_name, prename="val", check_valid=True)

    # logging & tracking
    #     1. prepare folder
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    output_path = f"./train_output/{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{config.exp_name}"

    os.makedirs(output_path, exist_ok=True)
    shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    #     2. preprare tracker
    writer = SummaryWriter(log_dir=f"{output_path}/tensorboard")

    # log values
    log_config(
        writer,
        epochs=epochs, 
        batch_size=batch_size,
        data_name=data_name, 
        data_path=data_path, 
        experiment_name=experiment_name,
        training_in_2d=training_in_2d, 
        sample_required_manhole_points=sample_required_manhole_points, 
        amount_non_manhole_samples=amount_non_manhole_samples,
        model_check_point_path=model_check_point_path,
        model_name=model_name,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        criterion_name=criterion_name,
        scheduler_name=scheduler_name,
        metrics_aggregator_name=metrics_aggregator_name,
        config=config_for_logging
    )

    # train loop
    global_step = 0
    for cur_epoch in tqdm(range(1, epochs+1), desc="MCR Training", total=epochs):

        train_metrics_aggregator.epoch_start(cur_epoch)
        model.train()

        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_metrics_aggregator.append(
                loss=loss.item()/inputs.size(0),
                inputs=inputs.detach().cpu(),
                labels=labels.detach().cpu(),
                preds=outputs.detach().cpu()
            )

            # save metrics in tensorboard
            for metric_name, metric_value in train_metrics_aggregator.get_cur_metrics().items():
                writer.add_scalar(f"{metric_name}/train", metric_value, global_step)

            global_step += 1

        scheduler.step()
        train_metrics_aggregator.epoch_end()

        val_metrics_aggregator.epoch_start(cur_epoch)
        model.eval()

        sample_amount = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_metrics_aggregator.append(
                    loss=loss.item()/inputs.size(0),
                    inputs=inputs.detach().cpu(),
                    labels=labels.detach().cpu(),
                    preds=outputs.detach().cpu()
                )

                # save metrics in tensorboard
                for metric_name, metric_value in val_metrics_aggregator.get_cur_metrics().items():
                    writer.add_scalar(f"{metric_name}/val", metric_value, cur_epoch)

                 # plot sample
                if sample_amount < 5:
                    # FIXME -> maybe have to apply sigmoid or softmax on outputs/preds
                    plot_2d_training_samples(input_img=inputs, label_img=labels, pred_img=outputs, 
                    title=f"Val Sample Epoch {cur_epoch}", 
                    save_path=f"{output_path}/sample_val_epoch_{cur_epoch}_{sample_amount}")
                    sample_amount += 1


        val_metrics_aggregator.epoch_end(cur_epoch)

        # save model checkpoints/weights
        if val_metrics_aggregator.have_new_best_metric():
            torch.save(model.state_dict(), f"{output_path}/best_model.pth")
        torch.save(model.state_dict(), f"{output_path}/latest_model.pth")

        # end of epoch print
        train_results = train_metrics_aggregator.get_metrics(with_prename=True)
        metrics_str = ""
        for key, value in train_results.items():
            metrics_str += f"{key}: {value:.2f} | "
        val_results = val_metrics_aggregator.get_metrics(with_prename=True)
        for key, value in val_results.items():
            metrics_str += f"{key}: {value:.2f} | "
        print(f"End of Epoch {cur_epoch} ({(cur_epoch/epochs)*100:.2f}%) | {metrics_str}")



# ---------------
# > Entry Point <
# ---------------
def train(config):
    batch_size = config.custom_train.batch_size
    epochs = config.custom_train.epochs
    learning_rate = config.custom_train.learning_rate
    optimizer = config.custom_train.optimizer

    model_name = config.model.name
    check_point_path = config.model.check_point_path

    data_name = config.data.name
    data_path = config.data.path
    data_preprocessed = config.data.preprocessed

    metrics_aggregator_name = config.custom_train.metrics_aggregator
    training_in_2d = config.custom_train.training_in_2d
    experiment_name = config.custom_train.experiment_name
    
    # call train
    train_pipeline()







