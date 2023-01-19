from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import click
import wandb
import datetime
import strategies
import os
from pathlib import Path


def get_data_loaders(bs, train_set=None, val_set=None, test_set=None):
    loaders = []

    if train_set:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=bs, shuffle=True, num_workers=8
        )
        loaders.append(train_loader)

    if val_set:
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=bs, shuffle=False, num_workers=8
        )
        loaders.append(val_loader)

    if test_set:
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=bs, shuffle=False, num_workers=8
        )
        loaders.append(test_loader)

    if len(loaders) == 1:
        return loaders[0]

    return tuple(loaders)


# fmt: off
@click.command()
@click.option("--epochs", default=10, help="Number of epochs to train per cycle.")
@click.option("--maxcycles", default=5, help="Number of active learning cycles to train.")
@click.option("--init_set_size", default=0.05, help="Initial train set size in percent.")
@click.option("--new_data_size", default=0.05, help="Percentage of added labels per cycle.")
@click.option("--lr", default=0.0001, help="Learning rate of the optimizer.")
@click.option("--bs", default=4, help="Batch size.")
@click.option("--sample_method", default="random", help="Sampling method to retrieve more labels.")
@click.option("--device", default="cuda", help="Device to train on.")
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
    maxcycles: int,
    init_set_size: float,
    new_data_size: float,
    lr: float,
    bs: int,
    sample_method: str,
    device: str,
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
        "maxcycles": maxcycles,
        "init_set_size": init_set_size,
        "new_data_size": new_data_size,
        "lr": lr,
        "bs": bs,
        "sample_method": sample_method,
        "device": device,
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

    print("Masking dataset...")

    # generate a random mask for the initial train set
    train_set.set_empty_mask()
    inital_elements = int(train_set.max_length() * init_set_size)
    train_set.add_random_labels(inital_elements)

    print("Loading model...")

    # load the model
    model = ImageCaptioningSystem(lr)

    cycle = 0
    # generate a string in the form of day-month-year-hour-minute for naming the wandb group
    date_time_str = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    group_name = f"ddp-{run_name}-{date_time_str}"
    run_name = f"{run_name}-{cycle}"

    wandb_logger = WandbLogger(
        project="active_learning",
        config=config,
        name=run_name,
        group=group_name,
    )

    print("Setup Trainer ...")
    if debug:
        trainer = pl.Trainer(
            accelerator=device,
            devices=num_devices,
            strategy="ddp",
            num_nodes=num_nodes,
            max_epochs=epochs,
            limit_train_batches=32,
            limit_val_batches=32,
            logger=wandb_logger,
            log_every_n_steps=8,
        )
    else:
        trainer = pl.Trainer(
            accelerator=device,
            devices=num_devices,
            strategy="ddp",
            num_nodes=num_nodes,
            d=epochs,
            val_check_interval=val_check_interval,
            logger=wandb_logger,
        )

    for cycle in range(maxcycles):
        print("Get Dataloaders ...")
        # train_loader, val_loader = get_data_loaders(bs, train_set, val_set)

        # print("Fit model ...")
        # trainer.fit(
        #     model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=val_loader,
        #     ckpt_path=ckpt_path,
        # )
        # trainer.save_checkpoint(
        #     f"/scratch/activelearning-ic/checkpoints/{run_name}-{date_time_str}-{cycle}.ckpt"
        # )

        elements_to_add = int(train_set.max_length() * new_data_size)

        if sample_method == "random":
            train_set.add_random_labels(elements_to_add)

        if sample_method == "cluster":
            train_set.flip_mask()
            train_loader = get_data_loaders(bs, train_set)
            clusters = strategies.image_diversity(train_loader, model, elements_to_add)
            train_set.flip_mask()


if __name__ == "__main__":
    train()
