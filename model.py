from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
import wandb
from torchmetrics import BLEUScore, MetricCollection, SacreBLEUScore, CHRFScore
from transformers import ViTFeatureExtractor
import pandas as pd
import functools


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

        metrics = MetricCollection([BLEUScore(), SacreBLEUScore(), CHRFScore()])

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

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
        

        for i in range(1,5):
            label = tokens[:, i, :].squeeze().long().reshape(-1)
            _loss = self.cross_entropy(inputs, label)
            loss += _loss

        return loss, logits

    def training_step(self, batch, batch_idx):
        """Calc

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
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
        # inference
        # label = sentences_token[:, 0, :].squeeze().long().contiguous()
        # output = self.model(pixel_values=pixel_values, labels=label)
        # logits = output.logits
        # output = self.model.generate(
        #     pixel_values=pixel_values, output_scores=True, return_dict_in_generate=True,
        #     min_length=57, max_length=57
        # )
        # logits = torch.stack(output.scores, axis=1)

        # # add loss for the five reference sentences
        # loss = [
        #     self.calculate_loss(logits, sentences_token[:, i, :].squeeze())
        #     for i in range(5)
        # ]

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        # calculate metrics
        for i in range(len(captions)):
            self.train_metrics([captions[i]], [sentences_text[i]])

        # calculate metrics score
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()

        # log metrics score by moving them to the cpu and converting them to python float
        for key in metrics:
            wandb.log({key: metrics[key].cpu().item()}, step=self.global_step)

        # log loss
        wandb.log({"train/loss": loss}, step=self.global_step)

        # log some examples
        if batch_idx % 100 == 0:
            data = {
                "epoch": [self.current_epoch] * batch_size,
                "step": [self.global_step] * batch_size,
                "truth": sentences_text,
                "prediction": captions,
            }
            self.train_examples = pd.concat(
                [self.train_examples, pd.DataFrame(data=data)]
            )
            wandb.log({"train/examples": self.train_examples})

        return loss

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

        # inference
        with torch.no_grad():
            pixel_values = pixel_values.squeeze()
            outputs = self.model.generate(pixel_values)

        # generate human readable captions
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # calculate metrics
        for i in range(len(captions)):
            self.val_metrics([captions[i]], [sentences_text[i]])

        # log some examples to wandb
        if batch_idx % 5 == 0:
            data = {
                "epoch": [self.current_epoch] * batch_size,
                "truth": sentences_text,
                "prediction": captions,
            }
            self.val_examples = pd.concat([self.val_examples, pd.DataFrame(data=data)])

    def validation_epoch_end(self, outputs):
        """_summary_

        Args:
            outputs (_type_): _description_
        """
        # calculate metrics score
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()

        for key in metrics:
            wandb.log({key: metrics[key].cpu().item()}, step=self.global_step)

        # log metrics and examples to wandb
        wandb.log({"val/examples": self.val_examples})

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
