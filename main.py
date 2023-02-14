import click

from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl

# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from prediction_writer import PredictionWriter
import torch

# from torch.utils.data import DataLoader
# import wandb
import datetime
import strategies
import os
from pathlib import Path

# from sklearn.model_selection import train_test_split
# from evaluation import eval_validation


# with open("secrets.txt", "r") as config_key:
#     api_key = config_key.readline().strip()


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
@click.option("--mode", default="train", help="Choose between train and test mode.")
@click.option("--seed", default=42, help="Global random seed.")
@click.option("--conf_mode", default="least", help="Whether to sample based on \"least\" confidence or \"margin\" of confidence")
@click.option("--conf_average", default="sentence", help="Whether to sample based on average \"sentence\" confidence or minimum \"word\" confidence")
@click.option("--cluster_mode", default="image", help="Whether to use the image or text embeddings for clustering")
@click.option("--mutliple_sentence_loss", is_flag=True, help="Whether to use the image or text embeddings for clustering")
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
    mode: str,
    seed: int,
    conf_mode: str,
    conf_average: str,
    cluster_mode: str,
    mutliple_sentence_loss: bool,
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
        "mode": mode,
        "seed": seed,
        "conf_mode": conf_mode,
        "conf_average": conf_average,
        "mutliple_sentence_loss": mutliple_sentence_loss,
    }

    # seed everything for reproducibility
    # between the different nodes and devices
    pl.seed_everything(seed)
    # set tokenizer parrallelism to false
    # this is needed because of the multiprocessing inherent to ddp
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # generate the correct paths for the images and the annotation json
    if device_type == "cuda":
        images_path = Path(data_path, "NWPU_images")
    else:
        images_path = Path(data_path)
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

    # print("Masking dataset...")
    # generate a random mask for the initial train set
    train_set.set_empty_mask()
    initial_elements = int(train_set.max_length() * init_set_size)
    train_set.add_random_labels(initial_elements)

    # generate a string in the form of day-month-year-hour-minute for naming the wandb group
    date_time_str = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    group_name = f"{run_name}-{date_time_str}"

    prediction_path_root = Path("predictions", run_name, date_time_str)
    prediction_path_root.mkdir(parents=True, exist_ok=True)

    strategy = "ddp" if num_devices > 1 or num_nodes > 1 else None
    limit_train_batches = 10 if debug else None
    limit_val_batches = 10 if debug else None
    log_every_n_steps = 8  # if debug else 50

    num_gpus = num_devices * num_nodes

    for cycle in range(max_cycles):
        print(f"----- CYCLE {cycle} -----")
        early_stopping_callback = EarlyStopping(monitor="val/loss_epoch", mode="min")
        prediction_writer = PredictionWriter(
            write_interval="epoch",
            root_dir=str(prediction_path_root),
            strategy=sample_method,
        )

        # wandb_run_name = f"{run_name}-{cycle}"
        # wandb.login(key=api_key)

        # if debug:
        #     wandb_logger = WandbLogger(
        #         mode="disabled",
        #         project="active_learning",
        #       config=config,
        #         name=wandb_run_name,
        #         group=group_name,
        #     )

        # if True:
        #     wandb_logger = WandbLogger(
        #         project="active_learning",
        #         config=config,
        #         name=wandb_run_name,
        #         group=group_name,
        #     )

        # print("Setup Trainer ...")
        trainer = pl.Trainer(
            callbacks=[prediction_writer],  # , early_stopping_callback],
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
            # logger=wandb_logger,
            num_sanity_val_steps=0,
        )

        # print("Loading model...")
        model = ImageCaptioningSystem(
            learning_rate, device_type, sample_method, mutliple_sentence_loss
        )
        prediction_writer.update_cycle(cycle)

        print("Get Dataloaders ...")
        train_loader, val_loader = get_data_loaders(
            batch_size, train_set=train_set, val_set=val_set
        )

        print(f"Fit model on {len(train_set)} samples...")
        if not debug:
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path,
            )

        if not debug:
            trainer.save_checkpoint(
                f"/home/users/w/wallburg/checkpoints/{run_name}-{date_time_str}-{cycle}.ckpt"
            )

        prediction_writer.update_mode("val")
        trainer.predict(model, val_loader)
        # wandb_logger.experiment.finish()

        if cycle == max_cycles - 1:
            break

        elements_to_add = int(train_set.max_length() * new_data_size)

        if sample_method == "random":
            print("Adding random labels ...")
            train_set.add_random_labels(elements_to_add)
            continue

        print("Predicting unlabeled data ...")
        prediction_writer.update_mode("unlabeled")

        train_set.flip_mask()
        unlabeled_loader = get_data_loaders(batch_size, unlabeled_set=train_set)
        trainer.predict(model, unlabeled_loader)
        train_set.flip_mask()

        unlabeled_prediction_path = prediction_writer.current_dir

        if sample_method == "confidence":
            img_ids = strategies.confidence_sample(
                path=unlabeled_prediction_path,
                elems_to_add=elements_to_add,
                mode=conf_mode,
                average=conf_average,
            )

        elif sample_method == "cluster":
            img_ids = strategies.diversity_based_sample(
                path=unlabeled_prediction_path,
                num_clusters=elements_to_add,
                type=cluster_mode,
                expected_num_files=num_gpus,
            )

        elif sample_method == "confidence+cluster":
            img_ids = strategies.conf_and_cluster(
                path=unlabeled_prediction_path,
                elems_to_add=elements_to_add,
                expected_num_files=num_gpus,
                type=cluster_mode,
                mode=conf_mode,
                conf_average=conf_average,
            )

        elif sample_method == "cluster+confidence":
            img_ids = strategies.cluster_and_conf(
                path=unlabeled_prediction_path,
                elems_to_add=elements_to_add,
                expected_num_files=num_gpus,
                type=cluster_mode,
                mode=conf_mode,
                conf_average=conf_average,
            )

        train_set.add_labels_by_img_id(img_ids)


if __name__ == "__main__":
    train()
