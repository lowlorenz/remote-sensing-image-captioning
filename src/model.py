import random

import pandas as pd
import lightning.pytorch as pl
import torch
import torchmetrics
import transformers
from omegaconf import OmegaConf
# from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.optim import Adam
from torchmetrics.text import BLEUScore, ROUGEScore
from transformers import (AutoProcessor, BlipForConditionalGeneration,
                          BlipProcessor, GPT2TokenizerFast, LlamaConfig,
                          LlamaForCausalLM, LlamaTokenizer,
                          VisionEncoderDecoderModel, ViTConfig, ViTModel)

import wandb

import random
# from peft import LoraConfig, TaskType, get_peft_model


transformers.utils.logging.disable_progress_bar()


class ImageCaptioningSystem(pl.LightningModule):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        """_summary_
        """

        self.cfg = cfg

        if self.cfg.model.name == "blip":
            self.tokenizer = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).tokenizer

            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to("cuda")

        if self.cfg.model.name == "gpt2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )

            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning",
            )
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.model.config.pad_token_id)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.max_tokens = 120

        metrics = torchmetrics.MetricCollection(
            [BLEUScore(), ROUGEScore(rouge_keys="rougeL")]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.train_examples = pd.DataFrame(
            columns=["epoch", "step", "truth", "prediction"]
        )
        self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])


    def forward(self, x):
        return self.model(x)

    def step(self, image, label, attention_mask):
        """calculate loss for a sentence - similar to the implementation of the model
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L628
        """
        batch_size, num_sentences, *_ = label.shape 
        sentence_id = torch.randint(0, num_sentences, (batch_size,))
        sampled_label = label[torch.arange(batch_size), sentence_id]
        sampled_attention_mask = attention_mask[torch.arange(batch_size), sentence_id]
        
        if self.cfg.model.name == "blip":
            prompt = self.tokenizer(
                "a photography of",
                return_tensors="pt",
                max_length=120,
                padding="max_length",
            ).input_ids

            output = self.model(
                pixel_values=image,
                input_ids=prompt,
                labels=sampled_label,
                attention_mask=sampled_attention_mask,
            )
        else:
            output = self.model(
                pixel_values=image,
                labels=sampled_label,
                decoder_attention_mask = sampled_attention_mask,
            )

        if self.cfg.model.name == "blip":
            logits = output.decoder_logits
        else:
            logits = output.logits

        return output.loss, logits

    def training_step(self, batch, batch_idx):
        """Fits the model to a batch of data and returns the loss

        Args:
            batch: Batch of the format (pixel_values, sentences_token, img_id, sentences_ids)
            batch_idx: Number of the batch

        Returns:
            _type_: CrossEntropyLoss of the batch
        """
        # prepare inputs
        image, label, attention_mask = batch

        # inference
        loss, logits = self.step(image, label, attention_mask)

        self.log("train_loss", loss, on_step=True, on_epoch=True)  # , sync_dist=True)

        if not self.cfg.compute.logging:
            return loss

        self.logging(image, label, logits, "train")

        return loss

    def on_training_epoch_end(self):
        wandb.log({'train examples': wandb.Table(dataframe=self.val_examples)})
        
    def validation_step(self, batch, batch_idx):
        # prepare inputs
        image, label, attention_mask = batch

        # inference
        # with torch.no_grad():
        loss, logits = self.step(image, label, attention_mask)

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.logging(image, label, logits, "val")

        return loss

    def on_validation_epoch_end(self):
        wandb.log({'val examples': wandb.Table(dataframe=self.val_examples)})

    def logging(self, image, label, logits, split):
        batch_size, num_sentences, num_tokens = label.shape 
        
        # `label` initially is of shape `(batch_size, num_sentences, num_tokens)`.
        # Reshaping it to `(batch_size * num_sentences, num_tokens)` ensures that each row 
        # of the resulting tensor represents a single sentence. This is necessary because the 
        # tokenizer's `batch_decode` method expects sequences (sentences) to be along the first dimension.
        label = label.reshape(batch_size * num_sentences, num_tokens)

        # Using the tokenizer's batch_decode method, we transform the token IDs back into human-readable text.
        # After decoding, `sentences_text` is a list where each entry corresponds to a decoded sentence.
        sentences_text = self.tokenizer.batch_decode(label, skip_special_tokens=True)

        # Now, we want to group the decoded sentences back according to their respective batches.
        # Each sublist inside `sentences_text` contains `num_sentences` sentences, which corresponds 
        # to the original layout before reshaping. 
        # So, for each batch `i`, we extract the sentences from position `i*num_sentences` to `(i+1)*num_sentences`.
        sentences_text = [sentences_text[i*num_sentences:(i+1)*num_sentences] for i in range(batch_size)]
         
        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True, 
        )

        data = {
            "epoch": [self.current_epoch] * batch_size,
            "step": [self.global_step] * batch_size,
            "truth": sentences_text,
            "prediction": captions,
        }
        
        if split == "train":
            logs = self.train_metrics(captions, sentences_text)
            self.train_examples = pd.concat([self.train_examples, pd.DataFrame(data=data)])
        elif split == "val":
            logs = self.val_metrics(captions, sentences_text)
            self.val_examples = pd.concat([self.val_examples, pd.DataFrame(data=data)])

        self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # if self.cfg.compute.strategy == "deepspeed_stage_2_offload":
        #     return DeepSpeedCPUAdam(self.parameters(), lr=self.cfg.training.lr)
        return Adam(self.parameters(), lr=self.cfg.training.lr)
