from dataset import NWPU_Captions, RSICD, Sydney_Captions, UCM_Captions
from model import ImageCaptioningSystem
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import torchvision.transforms as T
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from finetuning_scheduler import FinetuningScheduler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):    
    print(OmegaConf.to_yaml(cfg))
    # seed everything for reproducibility
    # between the different nodes and devices
    pl.seed_everything(cfg["seed"])
    # set tokenizer parrallelism to false
    # this is needed because of the multiprocessing inherent to ddp
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.info("Initalizing dataset...")

    dataset_lookup = {
        "NWPU-Captions": NWPU_Captions,
        "RSICD": RSICD,
        "Sydney-Captions": Sydney_Captions,
        "UCM-Captions": UCM_Captions,
    }
    
    transform = T.Compose([
    #     T.RandomHorizontalFlip(p=0.5),       # 1
        T.RandomVerticalFlip(p=0.5),         # 2
    #     # T.RandomRotation(degrees=15),        # 3
    #     # T.ColorJitter(brightness=0.2,        # 4
    #     #             contrast=0.2,
    #     #             saturation=0.2,
    #     #             hue=0.1),
    #     # T.RandomResizedCrop(224, scale=(0.8, 1.0)), # 5
    #     # T.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 6
    #     T.ToTensor()                         # 7
    ])

    
    Dataset = dataset_lookup[cfg.dataset.name]
    train_set = Dataset(
        cfg=cfg,
        split="train",
        transform=transform,
    )
    val_set = Dataset(
        cfg=cfg,
        split="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.compute.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.compute.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    logger = WandbLogger(
        cfg.dataset.name,
        project="image-captioning",
        # config=cfg,
    )

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{cfg.dataset.name} {cfg.model.name} {logger.experiment.id}",  # Save in a directory named after wandb run ID
        filename="{epoch}-{val_BLEUScore_epoch:.2f}",  # Save with filename containing epoch, step, and validation loss
        save_top_k=3,  # Save only the top 3 models based on the metric (in this case val_loss)
        verbose=True,  # Print when saving a checkpoint
        monitor="val_BLEUScore_epoch",  # Monitor validation loss for model saving
        mode="max",  # Save the model with minimum validation loss
    )

    early_stopping_callback = EarlyStopping(monitor="val_BLEUScore_epoch", mode="max", patience=8)

    # callbacks=[checkpoint_callback, FinetuningScheduler(ft_schedule="ImageCaptioningSystem_ft_schedule.yaml")],
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=cfg.compute.accelerator,
        num_nodes=cfg.compute.num_nodes,
        devices=cfg.compute.num_devices,
        accumulate_grad_batches=cfg.compute.accumulate_grad_batches,
        strategy=cfg.compute.strategy,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=16,
        num_sanity_val_steps=1,
        logger=logger,
        enable_progress_bar=False,
    )

    model = ImageCaptioningSystem(cfg)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
