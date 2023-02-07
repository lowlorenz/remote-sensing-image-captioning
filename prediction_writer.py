import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import os
from pathlib import Path


class PredictionWriter(BasePredictionWriter):
    def __init__(self, root_dir, write_interval, strategy):
        super().__init__(write_interval)
        self.root_dir = root_dir
        self.strategy = strategy
        self.cycle = 0
        self.mode = "unlabeled"

        self.current_dir = None
        self._update_path()

    def _update_path(self):
        self.current_dir = Path(self.root_dir, self.mode, str(self.cycle))
        self.current_dir.mkdir(parents=True, exist_ok=True)

    def update_cycle(self, cycle: int):
        self.cycle = cycle
        self._update_path()

    def update_mode(self, mode: str):
        mode = mode.lower()

        if mode not in ("train", "val", "test", "unlabeled"):
            raise ValueError("Mode should be one of 'train', 'val', 'test'")

        self.mode = mode
        self._update_path()

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # change format of list from
        # [(img_emb, img_id), (img_emb, img_id), ...] to
        # ([img_emb, img_emb, ...], [img_id, img_id, ...])
        # from https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
        confidences, margins, img_embeddings, img_ids, predicted_tokens = list(zip(*predictions[0]))
        img_ids = torch.cat(img_ids, axis=0)
        if "confidence" in self.strategy:
            confidences = torch.cat(confidences, axis=0)
            margins = torch.cat(margins, axis=0)
        if "cluster" in self.strategy:
            img_embeddings = torch.cat(img_embeddings, axis=0)
            predicted_tokens = torch.cat(predicted_tokens, axis=0)

        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        confidences_path = (self.current_dir/ f"confidences_rank_{trainer.global_rank}")
        margins_path = (self.current_dir/ f"margins_rank_{trainer.global_rank}")
        img_embedding_path = (
            self.current_dir / f"img_embeddings_rank_{trainer.global_rank}"
        )
        img_ids_path = self.current_dir / f"img_ids_rank_{trainer.global_rank}"
        predicted_tokens_path = (
            self.current_dir / f"predicted_tokens_rank_{trainer.global_rank}"
        )

        torch.save(confidences, str(confidences_path))
        torch.save(margins, str(margins_path))
        torch.save(img_embeddings, str(img_embedding_path))
        torch.save(img_ids, str(img_ids_path))
        torch.save(predicted_tokens, str(predicted_tokens_path))
