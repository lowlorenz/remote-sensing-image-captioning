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

# download('punkt')
# download('wordnet')
# download('omw-1.4')


class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr, device_type: str, sampling_method):
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

        self.train_examples = pd.DataFrame(
            columns=["epoch", "step", "truth", "prediction"]
        )
        self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.train_bleu = []
        self.val_bleu = []
        self.train_rouge = ROUGEScore()
        self.val_rouge = ROUGEScore()
        self.train_meteor = []
        self.val_meteor = []
        self.chencherry = bleu_score.SmoothingFunction()
        self.max_tokens = 56

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, pixel_values, tokens):
        """calculate loss for a sentence - similar to the implementation of the model
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L628
        """
        label = tokens[:, 0, :].long().contiguous()
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

        pixel_values = pixel_values.squeeze(dim=1)

        loss, logits = self.calculate_loss(pixel_values, sentences_token)

        self.log("train/loss", loss, on_step=True, on_epoch=False)

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        for i in range(len(captions)):
            self.train_bleu.append(bleu_score.sentence_bleu(sentences_text[i], captions[i][0]))
            # self.train_bleu.append(bleu_score.sentence_bleu([word_tokenize(e) for e in sentences_text[i]], word_tokenize(captions[i]), smoothing_function=self.chencherry.method1))
            self.train_rouge(captions[i], [sentences_text[i]])
            self.train_meteor.append(meteor([word_tokenize(e) for e in sentences_text[i]], word_tokenize(captions[i])))

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

        train_bleu = stats.mean(self.train_bleu)
        self.log("train/bleu", train_bleu, on_epoch=True)
        train_rouge = self.train_rouge.compute()
        self.log("train/rouge", train_rouge, on_epoch=True)
        train_meteor = stats.mean(self.train_meteor)
        self.log("train/meteor", train_meteor, on_epoch=True)

        self.logger.log_text(
            key="examples/train", dataframe=self.train_examples, step=self.global_step
        )

        self.train_meteor = []
        self.train_rouge.reset()
        self.train_bleu = []

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

        self.log("val/loss", loss, on_step=True, on_epoch=True)

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        for i in range(len(captions)):
            self.val_bleu.append(bleu_score.sentence_bleu(sentences_text[i], captions[i][0]))
            # self.val_bleu.append(bleu_score.sentence_bleu([word_tokenize(e) for e in sentences_text[i]], word_tokenize(captions[i]), smoothing_function=self.chencherry.method1))
            self.val_rouge(captions[i], [sentences_text[i]])
            self.val_meteor.append(meteor([word_tokenize(e) for e in sentences_text[i]], word_tokenize(captions[i])))

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
        val_bleu = self.val_bleu.compute()
        self.log("val/bleu", val_bleu, on_epoch=True)
        self.val_bleu.reset()
        val_rouge = self.val_rouge.compute()
        self.log("val/rouge", val_rouge, on_epoch=True)
        self.val_rouge.reset()
        val_meteor = stats.mean(self.val_meteor)
        self.log("val/meteor", val_meteor, on_epoch=True)
        self.val_meteor = []
        self.logger.log_text(
            key="examples/val", dataframe=self.val_examples, step=self.global_step
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        sentence_conf = None
        sentence_margin = None
        image_embeddings = None
        predicted_tokens = None
        pixel_values, sentences_token, img_ids, sentences_ids = batch
        pixel_values = pixel_values.squeeze(dim=1)
        bs = pixel_values.shape[0]
        label = sentences_token[:, 0, :].long().contiguous()

        with torch.no_grad():
            # Confidence
            if "conf" in self.method:
                # Öeast confidence
                out = self.model(pixel_values=pixel_values, labels=label, output_hidden_states=True)
                logits = out.logits
                logits_softmax = torch.nn.functional.softmax(logits, dim=2)
                word_conf, _ = torch.max(logits_softmax, dim=2)
                sentence_conf = torch.mean(word_conf, dim=1)
                # Margin of confidence
                top_2, _ = torch.topk(logits_softmax, 2, dim=2)
                word_margin = top_2[:, :, 0] - top_2[:, :, 1]
                sentence_margin = torch.mean(word_margin, dim=1)
                assert torch.numel(sentence_margin) == bs
            # Image diversity
            if "cluster" in self.method:
                image_embeddings = self.model.encoder(
                    pixel_values
                ).pooler_output  # (batch_size, hidden_size)
                logits = self.model(pixel_values=pixel_values, labels=label).logits

                predicted_tokens = logits.argmax(dim=-1)

        return sentence_conf, sentence_margin, image_embeddings, img_ids, predicted_tokens

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
