import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import os
from pathlib import Path


class PredictionWriter(BasePredictionWriter):
    def __init__(self, root_dir, write_interval):
        super().__init__(write_interval)
        self.root_dir = root_dir
        self.cycle = 0
        self._update_path()

    def _update_path(self):
        self.current_dir = Path(self.root_dir, str(self.cycle))
        self.current_dir.mkdir(parents=True, exist_ok=True)

    def update_cycle(self, cycle):
        self.cycle = cycle
        self._update_path()

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # change format of list from
        # [(img_emb, img_id), (img_emb, img_id), ...] to
        # ([img_emb, img_emb, ...], [img_id, img_id, ...])
        # from https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
        img_embeddings, img_ids = list(zip(*predictions[0]))
        img_embeddings = torch.cat(img_embeddings, axis=0)
        img_ids = torch.cat(img_ids, axis=0)

        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        img_embedding_path = (
            self.current_dir / f"img_embeddings_rank_{trainer.global_rank}"
        )
        img_ids_path = self.current_dir / f"img_ids_rank_{trainer.global_rank}"

        torch.save(
            img_embeddings,
            str(img_embedding_path),
        )
        torch.save(img_embeddings, str(img_ids_path))
