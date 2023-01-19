import torch
from model import ImageCaptioningSystem

def image_diversity(
    train_loader:torch.utils.data.DataLoader,
    model:ImageCaptioningSystem,
    elements_to_add:int
    ):
    

    image_embeddings = []
    # Get the embeddings of the images
    
    
    print(image_embeddings.shape)
            
        