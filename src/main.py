from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import os
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # seed everything for reproducibility
    # between the different nodes and devices
    pl.seed_everything(cfg["seed"])
    # set tokenizer parrallelism to false
    # this is needed because of the multiprocessing inherent to ddp
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    images_path = Path(cfg["data_path"], "NWPU_images")
    annotations_path = Path(cfg["data_path"], "dataset_nwpu.json")

    logging.info("Initalizing dataset...")

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
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        num_workers=cfg.training.num_workers
    )

    logger = WandbLogger(
        "NWPU-Captions",
        project="NWPU-Captions",
        config=cfg,
    )

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{logger.experiment.id}",  # Save in a directory named after wandb run ID
        filename="{epoch}-{step}-{val_loss:.2f}",  # Save with filename containing epoch, step, and validation loss
        save_top_k=3,  # Save only the top 3 models based on the metric (in this case val_loss)
        verbose=True,  # Print when saving a checkpoint
        monitor="val_loss",  # Monitor validation loss for model saving
        mode="min",  # Save the model with minimum validation loss
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu",
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=16,
        num_sanity_val_steps=1,
        logger=logger,
        enable_progress_bar=False,
        num_nodes=1,
        devices=cfg.training.num_devices,
        strategy="deepspeed_stage_2",
    )

    model = ImageCaptioningSystem(cfg.training.lr, cfg.training.mutliple_sentence_loss)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
