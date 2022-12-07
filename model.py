from transformers import VisionEncoderDecoderModel, BertTokenizer
import pytorch_lightning as pl
import torch

class ImageCaptioningSystem(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "bert-base-uncased"
        )
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.model.config.decoder_start_token_id = self.bert_tokenizer.cls_token_id
        self.model.config.pad_token_id = self.bert_tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pixel_values, sentences, _ = batch

        pixel_values = pixel_values.squeeze()        
        tokens = self.bert_tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

        outputs = self.model(pixel_values, labels=tokens)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, sentences, _ = batch

        pixel_values = pixel_values.squeeze()        
        tokens = self.bert_tokenizer(sentences, return_tensors = 'pt', padding='longest').input_ids.to(self.device)

        outputs = self.forward(pixel_values, labels=tokens)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)