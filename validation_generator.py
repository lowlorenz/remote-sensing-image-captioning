import click
from pathlib import Path
from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
import torch

from model import ImageCaptioningSystem
import pytorch_lightning as pl
from prediction_writer import PredictionWriter


@click.command()
@click.option(
    "--data_path", default="NWPU-Captions/", help="Path to the NWPU-Captions dataset."
)
@click.option("--num_cycles", default=1, help="Number of cycles to run.")
@click.argument("ckpt", type=click.Path(exists=True))
def generate_validation(data_path, num_cycles, ckpt):

    data_path = "NWPU-Captions/"
    images_path = Path(data_path, "NWPU_images")
    annotations_path = Path(data_path, "dataset_nwpu.json")
    val_set = NWPU_Captions(
        root=images_path,
        annotations_file=annotations_path,
        split="val",
        transform=ToTensor(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=12, shuffle=False, num_workers=4
    )

    # ckpt of form: conf_least_2-08-02-2023-16-46-0.ckpt = name_seed-time-cycle.ckpt
    ckpt_root = Path(ckpt).parent

    run_id = ckpt.split("/")[-1].split(".")[0]
    seed = run_id.split("_")[-1].split("-")[0]

    name_segment = run_id.split("_")[:-1]
    name = "_".join(name_segment)

    time_segments = run_id.split("-")[1:-1]
    time = "-".join(time_segments)

    prediction_path_root = Path("predictions", f"{name}_{seed}", time)
    prediction_path_root.mkdir(parents=True, exist_ok=True)

    prediction_writer = PredictionWriter(
        write_interval="epoch",
        root_dir=str(prediction_path_root),
        strategy="confidence",
    )
    prediction_writer.update_mode("val")

    for cycle in range(num_cycles):
        ckpt = ckpt_root / f"{name}_{seed}-{time}-{cycle}.ckpt"

        trainer = pl.Trainer(
            callbacks=[prediction_writer],
            accelerator="cuda",
        )

        model = ImageCaptioningSystem.load_from_checkpoint(
            ckpt,
            lr=0.001,
            device_type="cuda",
            sampling_method="confidence",
            mutliple_sentence_loss=False,
        )

        prediction_writer.update_cycle(cycle)
        trainer.predict(model, val_loader)


if __name__ == "__main__":
    generate_validation()
