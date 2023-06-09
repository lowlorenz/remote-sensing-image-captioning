from typing import Any, Tuple
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel
from torchmetrics import BLEUScore
from nltk.translate import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score as meteor
from nltk import word_tokenize, download
from transformers import ViTFeatureExtractor
import functools
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, List
import statistics as stats
import random


class ImageCaptioningSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        device_type: str,
        sampling_method: str,
        mutliple_sentence_loss: bool,
    ):
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

        self.device_type = device_type
        self.method = sampling_method
        self.mutliple_sentence_loss = mutliple_sentence_loss

        self.train_examples = pd.DataFrame(
            columns=["epoch", "step", "truth", "prediction"]
        )
        self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.max_tokens = 56

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, pixel_values, tokens):
        """calculate loss for a sentence - similar to the implementation of the model
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L628
        """
        if self.mutliple_sentence_loss:
            label = tokens[:, 0, :].long().contiguous()
            output = self.model(pixel_values=pixel_values, labels=label)

            logits = output.logits
            loss = output.loss

            inputs = logits.reshape(-1, self.model.decoder.config.vocab_size)

            for i in range(1, 5):
                label = tokens[:, i, :].squeeze().long().reshape(-1)
                _loss = self.cross_entropy(inputs, label)
                loss += _loss

        else:
            index = random.randint(0, 4)
            label = tokens[:, index, :].squeeze().long().contiguous()
            output = self.model(pixel_values=pixel_values, labels=label)
            loss = output.loss
            logits = output.logits

        return loss, logits

    def training_step(self, batch, batch_idx):
        """Fits the model to a batch of data and returns the loss

        Args:
            batch: Batch of the format (pixel_values, sentences_token, img_id, sentences_ids)
            batch_idx: Number of the batch

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

        pixel_values = pixel_values.squeeze(dim=1)

        loss, logits = self.calculate_loss(pixel_values, sentences_token)

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

    def validation_step(self, batch, batch_idx):
        # prepare inputs
        pixel_values, sentences_token, img_id, sentences_ids = batch
        batch_size = len(img_id)

        sentences_text = [
            self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        pixel_values = pixel_values.squeeze(dim=1)

        # inference
        with torch.no_grad():
            loss, logits = self.calculate_loss(pixel_values, sentences_token)

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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        sentence_conf = None
        sentence_margin = None
        least_word_conf = None
        least_word_margin = None
        image_embeddings = None
        predicted_tokens = None
        pixel_values, sentences_token, img_ids, sentences_ids = batch
        pixel_values = pixel_values.squeeze(dim=1)
        bs = pixel_values.shape[0]
        label = sentences_token[:, 0, :].long().contiguous()

        with torch.no_grad():
            # Confidence
            if "conf" in self.method:
                # Least confidence
                out = self.model(
                    pixel_values=pixel_values, labels=label, output_hidden_states=True
                )
                logits = out.logits
                logits_softmax = torch.nn.functional.softmax(logits, dim=2)
                word_conf, _ = torch.max(logits_softmax, dim=2)
                least_word_conf, _ = torch.min(word_conf, dim=1)
                sentence_conf = torch.mean(word_conf, dim=1)
                # Margin of confidence
                top_2, _ = torch.topk(logits_softmax, 2, dim=2)
                word_margin = top_2[:, :, 0] - top_2[:, :, 1]
                least_word_margin, _ = torch.min(word_margin, dim=1)
                sentence_margin = torch.mean(word_margin, dim=1)
                assert torch.numel(sentence_margin) == bs

            # Image diversity
            if "cluster" in self.method:
                # get the image embeddings by calling the encoder and retrieving only the pooler layer output
                image_embeddings = self.model.encoder(
                    pixel_values
                ).pooler_output
                # get the logits by calling the whole model on the image
                logits = self.model(pixel_values=pixel_values, labels=label).logits

            if "cluster" not in self.method and "conf" not in self.method:
                logits = self.model(pixel_values=pixel_values, labels=label).logits

            predicted_tokens = logits.argmax(dim=-1)

        return (
            sentence_conf,
            least_word_conf,
            sentence_margin,
            least_word_margin,
            image_embeddings,
            img_ids,
            predicted_tokens,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
