from transformers import (
    VisionEncoderDecoderModel,
    GPT2TokenizerFast,
    VisionEncoderDecoderModel,
    LlamaConfig,
    ViTConfig,
    LlamaForCausalLM,
    ViTModel,
    LlamaTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoTokenizer,
)
import transformers
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.optim import Adam
from peft import LoraConfig, TaskType, get_peft_model

import pandas as pd
import pytorch_lightning as pl
import torch
import random
from torchmetrics.text import BLEUScore, ROUGEScore
import torchmetrics
from omegaconf import OmegaConf

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

        if self.cfg.model.name == "gpt2":
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning",
            )
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.model.decoder.config.decoder_start_token_id = (
                self.model.decoder.config.bos_token_id
            )

        elif self.cfg.model.name == "llama2":
            encoder_config = AutoConfig.from_pretrained(self.cfg.model.encoder_path)
            decoder_config = AutoConfig.from_pretrained(self.cfg.model.decoder_path)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_pretrained_model_name_or_path=self.cfg.model.encoder_path, 
                decoder_pretrained_model_name_or_path=self.cfg.model.decoder_path,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
            )
            
            decoder_start_token_id = decoder_config.decoder_start_token_id
            pad_token_id = decoder_config.pad_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = decoder_config.bos_token_id
            if pad_token_id is None:
                pad_token_id = decoder_config.eos_token_id
            
            print(self.model.config)
            self.model.config.decoder.is_decoder = True
            self.model.config.eos_token_id = decoder_config.eos_token_id
            self.model.config.decoder_start_token_id = decoder_start_token_id
            self.model.config.pad_token_id = pad_token_id
                    
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model.decoder_path,
            )
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.model.config.pad_token_id)


        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.max_tokens = 120

        # config = LoraConfig(
        #     r=16,
        #     lora_alpha=16,
        #     target_modules=["query", "value"],
        #     lora_dropout=0.1,
        #     bias="none",
        #     modules_to_save=["classifier"],
        # )
        # self.model = get_peft_model(self.model, config)

        if self.cfg.compute.logging:
            self.train_examples = pd.DataFrame(
                columns=["epoch", "step", "truth", "prediction"]
            )
            self.val_examples = pd.DataFrame(columns=["epoch", "truth", "prediction"])

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
        if self.cfg.training.mutliple_sentence_loss:
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

        pixel_values = pixel_values.squeeze(dim=1)

        loss, logits = self.calculate_loss(pixel_values, sentences_token)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        if not self.cfg.compute.logging:

            return loss
        sentences_text = [
            self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        # detokenize human readable captions
        captions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )

        logs = self.train_metrics(captions, sentences_text)
        self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)

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

        # sentences_text = [
        #     self.tokenizer.batch_decode(sentences_token[i], skip_special_tokens=True)
        #     for i in range(batch_size)
        # ]

        pixel_values = pixel_values.squeeze(dim=1)

        # inference
        with torch.no_grad():
            loss, logits = self.calculate_loss(pixel_values, sentences_token)

        # detokenize human readable captions
        # captions = self.tokenizer.batch_decode(
        #     logits.argmax(dim=-1), skip_special_tokens=True
        # )

        # logs = self.val_metrics(captions, sentences_text)
        # self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

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
        if self.cfg.compute.strategy == "deepspeed_stage_2_offload":
            return DeepSpeedCPUAdam(self.parameters(), lr=self.cfg.training.lr)
        return Adam(self.parameters(), lr=self.cfg.training.lr)
