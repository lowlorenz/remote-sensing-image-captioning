from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
import pytorch_lightning as pl
import torch
from torchmetrics import BLEUScore

class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        self.lr = lr
        self.val_bleu = BLEUScore()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pixel_values, sentences, _ = batch

        pixel_values = pixel_values.squeeze()        
        tokens = self.tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

        outputs = self.model(pixel_values, labels=tokens)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pixel_values, sentences, _ = batch

            pixel_values = pixel_values.squeeze()        
            tokens = self.tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

            outputs = self.model(pixel_values, labels=tokens)
            
            loss = outputs.loss
            
        captions = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
        self.val_bleu(captions, sentences)

        return loss

    def validation_epoch_end(self, outputs):
        self.log('val/bleu', self.val_bleu.compute(), True)
        self.val_bleu.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)