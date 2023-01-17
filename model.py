from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
from torchmetrics import BLEUScore, MetricCollection, MeanMetric
# from torchmetrics.text.rouge import ROUGEScore
from transformers import ViTFeatureExtractor
import pandas as pd
import functools
from pytorch_lightning.utilities import rank_zero_only


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

        text_metrics = MetricCollection([BLEUScore()])
        self.train_text_metrics = text_metrics.clone(prefix="train/")
        self.val_text_metrics = text_metrics.clone(prefix="val/")

        self.train_loss = MeanMetric()

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
        """ Fits the model to a batch of data and returns the loss 

        Args:
            batch (_type_): Batch of the format (pixel_values, sentences_token, img_id, sentences_ids)
            batch_idx (_type_): Number of the batch

        Returns:
            _type_: CrossEntropyLoss of the batch
        """
        print("training step on rank: ", self.global_rank)
        # prepare inputs
        pixel_values, sentences_token, img_id, sentences_ids = batch
        batch_size = len(img_id)

        sentences_text = [
            self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        pixel_values = pixel_values.squeeze()

        loss, logits = self.calculate_loss(pixel_values, sentences_token)

        self.train_loss.update(loss)

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        # calculate metrics
        for i in range(len(captions)):
            self.train_text_metrics.update([captions[i]], [sentences_text[i]])

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # if this is not the main process, do not log examples
        if self.global_rank != 0:
            return loss
        
        # if this is the main process, log examples every 100 batches
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
            #self.log_text({"train/examples": self.train_examples})

        return loss
    
    def training_epoch_end(self, outputs):
        if self.global_rank != 0:
            return
        
        # calculate metrics score
        metrics = self.train_text_metrics.compute()
        self.train_text_metrics.reset()

        # log metrics score by moving them to the cpu and converting them to python float
        log_dict = {
            key: value.cpu().item() for key,value in metrics
        }

        # log accumulated loss
        self.train_loss.reset()
        loss = self.train_loss.compute()
        log_dict["train/loss"] = loss.cpu().item()

        # log epoch
        log_dict["epoch"] = self.current_epoch
        
        # log metrics
        self.log_dict(log_dict, step=self.global_step)


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
            outputs = self.model.generate(pixel_values, max_new_tokens=60)

        # generate human readable captions
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # calculate metrics
        for i in range(len(captions)):
            self.val_text_metrics.update([captions[i]], [sentences_text[i]])

        # if this is not the main process, do not log examples
        if self.global_rank != 0:
            return 
        
        # log some examples
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
        
        if self.global_rank != 0:
            return
                
        # calculate metrics score
        metrics = self.val_text_metrics.compute()
        self.val_text_metrics.reset()

        # log metrics score by moving them to the cpu and converting them to python float
        log_dict = {
            key: value.cpu().item() for key,value in metrics
        }

        # log epoch
        log_dict["epoch"] = self.current_epoch
        
        # log metrics
        self.log_dict(log_dict, step=self.global_step)


    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
