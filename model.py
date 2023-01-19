from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
from torchmetrics import BLEUScore, MetricCollection, MeanMetric

# from torchmetrics.text.rouge import ROUGEScore
from transformers import ViTFeatureExtractor
import pandas as pd
import functools
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, List


class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        """_summary_
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.lr = lr

        self.train_examples = pd.DataFrame(
            columns=["epoch", "step", "truth", "prediction"]
        )
        self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.model(x)

    def calculate_loss(self, pixel_values, tokens):
        """calculate loss for a sentence - similar to the implementation of the model
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L628

        Args:
            inputs (_type_): _description_
            targets (_type_): _description_

        Returns:
            _type_: _description_
        """
        label = tokens[:, 0, :].squeeze().long().contiguous()
        output = self.model(pixel_values=pixel_values, labels=label)

        logits = output.logits
        loss = output.loss

        inputs = logits.reshape(-1, self.model.decoder.config.vocab_size)

        for i in range(1, 5):
            label = tokens[:, i, :].squeeze().long().reshape(-1)
            _loss = self.cross_entropy(inputs, label)
            loss += _loss

        return loss, logits

    def training_step(self, batch, batch_idx):
        """Fits the model to a batch of data and returns the loss

        Args:
            batch (_type_): Batch of the format (pixel_values, sentences_token, img_id, sentences_ids)
            batch_idx (_type_): Number of the batch

        Returns:
            _type_: CrossEntropyLoss of the batch
        """
        # prepare inputs
        pixel_values, sentences_token, img_id, sentences_ids = batch
        batch_size = len(img_id)

        sentences_text = [
            self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        pixel_values = pixel_values.squeeze()

        loss, logits = self.calculate_loss(pixel_values, sentences_token)

        self.log("train/loss", loss, on_step=True, on_epoch=False)

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        # if this is not the main process, do not log examples
        if self.global_rank != 0:
            return loss

        # if this is the main process, log examples every 100 batches
        if batch_idx % 100 != 0:
            return loss

        data = {
            "epoch": [self.current_epoch] * batch_size,
            "step": [self.global_step] * batch_size,
            "truth": sentences_text,
            "prediction": captions,
        }

        self.train_examples = pd.concat([self.train_examples, pd.DataFrame(data=data)])

        return loss

    def training_epoch_end(self, outputs):
        if self.global_rank != 0:
            return

        self.logger.log_text(
            key="examples/train", dataframe=self.train_examples, step=self.global_step
        )

    def validation_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        # prepare inputs
        pixel_values, sentences_token, img_id, sentences_ids = batch
        batch_size = len(img_id)

        sentences_text = [
            self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        pixel_values = pixel_values.squeeze()

        # inference
        with torch.no_grad():
            loss, logits = self.calculate_loss(pixel_values, sentences_token)

        self.log("val/loss", loss, on_step=True, on_epoch=True)

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        # if this is not the main process, do not log examples
        if self.global_rank != 0:
            return

        # log some examples
        if batch_idx % 5 != 0:
            return

        data = {
            "epoch": [self.current_epoch] * batch_size,
            "step": [self.global_step] * batch_size,
            "truth": sentences_text,
            "prediction": captions,
        }

        self.val_examples = pd.concat([self.val_examples, pd.DataFrame(data=data)])

    def validation_epoch_end(self, outputs):

        if self.global_rank != 0:
            return

        self.logger.log_text(
            key="examples/val", dataframe=self.val_examples, step=self.global_step
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        pixel_values, sentences, img_ids, sentences_ids = batch
        pixel_values = pixel_values.squeeze()

        with torch.no_grad():
            image_embeddings = self.model.encoder(
                pixel_values
            ).last_hidden_state  # (bs, 196, 768)

        return image_embeddings, img_ids

    # def on_predict_epoch_end(self, results: List[Any]) -> None:

    #     # change format of list from
    #     # [(img_emb, img_id), (img_emb, img_id), ...] to
    #     # ([img_emb, img_emb, ...], [img_id, img_id, ...])
    #     # from https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
    #     img_embeddings, img_ids = list(zip(*results))
    #     img_embeddings = torch.cat(img_embeddings, axis=0)
    #     img_ids = torch.cat(img_ids, axis=0)

    #     return img_embeddings, img_ids

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
