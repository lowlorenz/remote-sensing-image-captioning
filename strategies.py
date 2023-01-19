import torch
from model import ImageCaptioningSystem

def image_diversity(train_loader:torch.utils.data.DataLoader, model:ImageCaptioningSystem, elements_to_add:int):
    # Get the embeddings of the images
    for batch in train_loader:
        pixel_values, sentences, img_ids, sentences_ids = batch
        pixel_values = pixel_values.squeeze()
        with torch.no_grad():
            image_embeddings = model.model.encoder(pixel_values)
            
        