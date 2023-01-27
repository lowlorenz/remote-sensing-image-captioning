from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from prediction_writer import PredictionWriter
import torch
from torch.utils.data import Subset
import click
import datetime
import strategies
import os
from pathlib import Path
from evaluation import eval_validation


def get_data_loaders(
    batch_size, train_set=None, val_set=None, test_set=None, unlabeled_set=None
):
    loaders = []

    if train_set:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4
        )
        loaders.append(train_loader)

    if val_set:
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=4
        )
        loaders.append(val_loader)

    if test_set:
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=4
        )
        loaders.append(test_loader)

    if unlabeled_set:
        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=4
        )
        loaders.append(unlabeled_loader)

    if len(loaders) == 1:
        return loaders[0]

    return tuple(loaders)


def build_lightning_trainer(**kwargs):
    prediction_writer = PredictionWriter(
        write_interval="epoch", root_dir=str(kwargs["prediction_path_root"])
    )
    wandb_run_name = f"{kwargs['run_name']}-{kwargs['cycle']}"

    wandb_logger = WandbLogger(
        project="active_learning",
        config=kwargs["config"],
        name=kwargs["wandb_run_name"],
        group=kwargs["group_name"],
    )

    print("Setup Trainer ...")
    trainer = pl.Trainer(
        callbacks=[prediction_writer],
        accelerator=kwargs["device_type"],
        devices=kwargs["num_devices"],
        strategy=kwargs["strategy"],
        num_nodes=kwargs["num_nodes"],
        max_epochs=kwargs["epochs"],
        val_check_interval=kwargs["val_check_interval"],
        limit_train_batches=kwargs["limit_train_batches"],
        limit_val_batches=kwargs["limit_val_batches"],
        log_every_n_steps=kwargs["log_every_n_steps"],
        precision=16,
        logger=wandb_logger,
    )

    return train, wandb_logger, prediction_writer


# fmt: off
@click.command()
@click.option("--epochs", default=10, help="Number of epochs to train per cycle.")
@click.option("--max_cycles", default=5, help="Number of active learning cycles to train.")
@click.option("--init_set_size", default=0.05, help="Initial train set size in percent.")
@click.option("--new_data_size", default=0.05, help="Percentage of added labels per cycle.")
@click.option("--learning_rate", default=0.0001, help="Learning rate of the optimizer.")
@click.option("--batch_size", default=4, help="Batch size.")
@click.option("--sample_method", default="cluster", help="Sampling method to retrieve more labels.")
@click.option("--device_type", default="cuda", help="Device to train on.")
@click.option("--run_name", default="test", help="Name of the run.")
@click.option("--data_path", default="NWPU-Captions/", help="Path to the NWPU-Captions dataset.")
@click.option("--debug", is_flag=True, help="Debug mode.")
@click.option("--val_check_interval", default=1.0, help="Validation check interval.")
@click.option("--num_devices", default=1, help="Number of devices to train on.")
@click.option("--num_nodes", default=1, help="Number of nodes to train on.")
@click.option("--ckpt_path", default=None, help="Path to checkpoint to resume training.")
# fmt: on
def train(
    epochs: int,
    max_cycles: int,
    init_set_size: float,
    new_data_size: float,
    learning_rate: float,
    batch_size: int,
    sample_method: str,
    device_type: str,
    run_name: str,
    data_path: str,
    debug: bool,
    val_check_interval: float,
    num_devices: int,
    num_nodes: int,
    ckpt_path: str,
) -> None:
    # save the config for wandb
    config = {
        "epochs": epochs,
        "max_cycles": max_cycles,
        "init_set_size": init_set_size,
        "new_data_size": new_data_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "sample_method": sample_method,
        "device_type": device_type,
        "run_name": run_name,
        "data_path": data_path,
        "debug": debug,
        "val_check_interval": val_check_interval,
        "num_devices": num_devices,
        "num_nodes": num_nodes,
        "ckpt_path": ckpt_path,
    }

    # seed everything for reproducibility
    # between the different nodes and devices
    pl.seed_everything(42)
    # set tokenizer parrallelism to false
    # this is needed because of the multiprocessing inherent to ddp
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # generate the correct paths for the images and the annotation json
    images_path = Path(data_path, "NWPU_images")
    annotations_path = Path(data_path, "dataset_nwpu.json")

    print("Initalizing dataset...")

    train_set = NWPU_Captions(
        root=images_path,
        annotations_file=annotations_path,
        split="train",
        transform=ToTensor(),
    )
    val_set = NWPU_Captions(
        root=images_path,
        annotations_file=annotations_path,
        split="val",
        transform=ToTensor(),
    )
    test_set = NWPU_Captions(
        root=images_path,
        annotations_file=annotations_path,
        split="test",
        transform=ToTensor(),
    )

    if debug:
        train_set.set_empty_mask()
        val_set.set_empty_mask()
        test_set.set_empty_mask()

        train_set.add_random_labels(100)
        val_set.add_random_labels(100)
        test_set.add_random_labels(100)

    print("Masking dataset...")

    # generate a random mask for the initial train set
    # train_set.set_empty_mask()
    # inital_elements = int(train_set.max_length() * init_set_size)
    # train_set.add_random_labels(inital_elements)

    # generate a string in the form of day-month-year-hour-minute for naming the wandb group
    date_time_str = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    group_name = f"{run_name}-{date_time_str}"

    prediction_path_root = Path("predictions", run_name, date_time_str)
    prediction_path_root.mkdir(parents=True, exist_ok=True)

    strategy = "ddp" if num_devices > 1 or num_nodes > 1 else None
    limit_train_batches = 32 if debug else None
    limit_val_batches = 32 if debug else None
    log_every_n_steps = 8  # if debug else 50

    for cycle in range(max_cycles):
        prediction_writer = PredictionWriter(
            write_interval="epoch", root_dir=str(prediction_path_root)
        )
        wandb_run_name = f"{run_name}-{cycle}"

        wandb_logger = WandbLogger(
            project="active_learning",
            config=config,
            name=wandb_run_name,
            group=group_name,
        )

        print("Setup Trainer ...")
        trainer = pl.Trainer(
            callbacks=[prediction_writer],
            accelerator=device_type,
            devices=num_devices,
            strategy=strategy,
            num_nodes=num_nodes,
            max_epochs=epochs,
            val_check_interval=val_check_interval,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            precision=16,
            logger=wandb_logger,
        )

        print("Loading model...")
        model = ImageCaptioningSystem(learning_rate)
        prediction_writer.update_cycle(cycle)

        print("Get Dataloaders ...")
        train_loader, val_loader = get_data_loaders(
            batch_size, train_set=train_set, val_set=val_set
        )

        print("Fit model ...")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        trainer.save_checkpoint(
            f"/scratch/activelearning-ic/checkpoints/{run_name}-{date_time_str}-{cycle}.ckpt"
        )

        prediction_writer.update_mode("val")
        trainer.predict(model, val_loader)

        val_prediction_path = prediction_writer.current_dir
        reference_paths = [f"val_references_{i}.txt" for i in range(5)]

        mean_bleu, mean_meteor = eval_validation(reference_paths, val_prediction_path)

        wandb_logger.experiment.summary["val_bleu"] = mean_bleu
        wandb_logger.experiment.summary["val_meteor"] = mean_meteor
        wandb_logger.experiment.summary["cycle"] = cycle

        wandb_logger.experiment.finish()

        if cycle == max_cycles - 1:
            break

        elements_to_add = int(train_set.max_length() * new_data_size)

        print("Predicting unlabeled data ...")
        prediction_writer.update_mode("unlabeled")

        train_set.flip_mask()
        unlabeled_loader = get_data_loaders(batch_size, unlabeled_set=train_set)
        trainer.predict(model, unlabeled_loader)
        train_set.flip_mask()

        unlabeled_prediction_path = prediction_writer.current_dir

        print("Adding new labels ...")
        if sample_method == "random":
            train_set.add_random_labels(elements_to_add)

        if sample_method == "cluster":

            img_ids = strategies.diversity_based_sample(
                unlabeled_prediction_path, elements_to_add, "image"
            )

            train_set.add_labels_by_img_id(img_ids)


if __name__ == "__main__":
    train()
