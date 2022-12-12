from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
import wandb
from torchmetrics import BLEUScore

class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        self.lr = lr
        self.train_bleu = BLEUScore()
        self.val_bleu = BLEUScore()
        self.table = wandb.Table(columns=['epoch', 'truth', 'prediction'])

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
        self.train_bleu(captions, sentences)    

        # log loss to wandb
        wandb.log({'train/loss': loss}, step=self.global_step)
        return loss

    def on_train_epoch_end(self, outputs):
        # calculate bleu score
        bleu = self.train_bleu.compute()
        self.train_bleu.reset()

        # log bleu score
        self.log('train/bleu', bleu, True)
        wandb.log({'train/bleu': bleu, 'epoch': self.current_epoch}, step=self.global_step)

    def validation_step(self, batch, batch_idx):
        # inference
        with torch.no_grad():
            pixel_values, sentences, _ = batch

            pixel_values = pixel_values.squeeze()        
            tokens = self.tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

            outputs = self.model(pixel_values, labels=tokens)
            
            loss = outputs.loss
            
        # generate human readable captions
        captions = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1))

        # calculate bleu score
        self.val_bleu(captions, sentences)

        # log some examples to wandb
        if batch_idx % 5 == 0:
            for i in range(len(sentences)):
                self.table.add_data(self.current_epoch, sentences[i], captions[i])
            wandb.log({"val/examples": self.table})

        return loss

    def on_validation_epoch_end(self, outputs):
        # calculate bleu score
        bleu = self.val_bleu.compute()
        self.val_bleu.reset()

        # log bleu score and loss to wandb
        self.log('val/bleu', bleu, True)
        wandb.log({'val/bleu': bleu, 'val/loss': torch.stack(outputs).mean(), 'epoch': self.current_epoch}, step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)