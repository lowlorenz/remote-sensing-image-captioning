from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
import wandb
from torchmetrics import BLEUScore,  MetricCollection, SacreBLEUScore, CHRFScore
from transformers import ViTFeatureExtractor
import pandas as pd

class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_processor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.lr = lr
        self.train_examples = pd.DataFrame(columns=['epoch', 'step', 'truth', 'prediction'])
        self.val_examples = pd.DataFrame(columns=['epoch', 'truth', 'prediction'])

        metrics = MetricCollection([
            BLEUScore(), SacreBLEUScore(), CHRFScore()
        ])
        
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # prepare inputs
        pixel_values, sentences, _ = batch

        pixel_values = pixel_values.squeeze()   
        tokens = self.tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

        # inference
        outputs = self.model(pixel_values, labels=tokens)
        loss = outputs.loss
        
        # generate human readable captions
        captions = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
        captions = [s[:s.find('<|endoftext|>')] for s in captions]

        # calculate metrics
        for i in range(len(captions)):            
            self.train_metrics([captions[i]], [[sentences[i]]])    
            
        # calculate metrics score
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()

        # log metrics score
        for key in metrics:           
            wandb.log({key:metrics[key].cpu().item()}, step=self.global_step)

        # log loss
        wandb.log({'train/loss': loss}, step=self.global_step)

        # log some examples
        if batch_idx % 100 == 0:
            data = {'epoch': [self.current_epoch] * len(sentences), 'step': [self.global_step] * len(sentences), 'truth': sentences, 'prediction': captions}
            self.train_examples = pd.concat([self.train_examples, pd.DataFrame(data=data)])
            wandb.log({"train/examples": self.train_examples})

        return loss


    def validation_step(self, batch, batch_idx):
        # inference
        with torch.no_grad():        
            # prepare inputs
            pixel_values, sentences, _ = batch
            
            pixel_values = pixel_values.squeeze()        
            tokens = self.tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

            outputs = self.model(pixel_values, labels=tokens)
            
            loss = outputs.loss
            
        # generate human readable captions
        captions = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
        captions = [s[:s.find('<|endoftext|>')] for s in captions]

        # calculate metrics
        for i in range(len(captions)):
            self.val_metrics([captions[i]], [[sentences[i]]])

        # log some examples to wandb
        if batch_idx % 5 == 0:
            data = {'epoch': [self.current_epoch] * len(sentences), 'truth': sentences, 'prediction': captions}
            self.val_examples = pd.concat([self.val_examples, pd.DataFrame(data=data)])
            
        wandb.log({'val/loss': loss}, step=self.global_step)
        return loss

    def validation_epoch_end(self, outputs):
        # calculate metrics score
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        
        for key in metrics:           
            wandb.log({key:metrics[key].cpu().item()}, step=self.global_step)

        # log metrics and examples to wandb
        wandb.log({"val/examples": self.val_examples})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)