from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
from peft import LoraConfig, TaskType, get_peft_model

import pandas as pd
import pytorch_lightning as pl
import torch
import random
from torchmetrics.text import BLEUScore, ROUGEScore
import torchmetrics


class ImageCaptioningSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        mutliple_sentence_loss: bool,
    ):
        super().__init__()
        """_summary_
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning",
        )

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model = get_peft_model(self.model, config)

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.lr = lr
        self.mutliple_sentence_loss = mutliple_sentence_loss

        self.train_examples = pd.DataFrame(
            columns=["epoch", "step", "truth", "prediction"]
        )
        self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.max_tokens = 56

        metrics = torchmetrics.MetricCollection(
            [BLEUScore(), ROUGEScore(rouge_keys="rougeL")]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

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

        logs = self.train_metrics(captions, sentences_text)
        self.log_dict(logs, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        # # if this is the main process, log examples every 100 batches
        # if batch_idx % 100 != 0:
        #     return loss

        # data = {
        #     "epoch": [self.current_epoch] * batch_size,
        #     "step": [self.global_step] * batch_size,
        #     "truth": sentences_text,
        #     "prediction": captions,
        # }

        # self.train_examples = pd.concat([self.train_examples, pd.DataFrame(data=data)])

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

        logs = self.val_metrics(captions, sentences_text)
        self.log_dict(logs, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        # # log some examples
        # if batch_idx % 5 != 0:
        #     return

        # data = {
        #     "epoch": [self.current_epoch] * batch_size,
        #     "step": [self.global_step] * batch_size,
        #     "truth": sentences_text,
        #     "prediction": captions,
        # }

        # print(data)
        # self.val_examples = pd.concat([self.val_examples, pd.DataFrame(data=data)])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
