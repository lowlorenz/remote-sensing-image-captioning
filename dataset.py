import torch
import torchvision.datasets as datasets
import PIL
import pandas as pd
import json
import unicodedata
import numpy as np
from pathlib import Path
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer
from torchvision.transforms import ToTensor

def read_annotations_file(root, annotation_file):
    file_content = open(annotation_file, encoding='utf8').read()
    file_content = unicodedata.normalize("NFKD", file_content)
    metadata = json.loads(file_content)

    datasets = {
        'train': {
                'image': [],
                'sentence': [],
                'identifier': [],
            },
        'val':  {
                'image': [],
                'sentence': [],
                'identifier': [],
            },
        'test':  {
                'image': [],
                'sentence': [],
                'identifier': [],
            }
    }
    
    # Scraping the data from the json file
    for k in list(metadata.keys()):
        for i in range(len(metadata[k])):
            ds = datasets[metadata[k][i]['split']]
            
            ds['image'].append(Path(root, k, metadata[k][i]['filename']))
            ds['image'].append(Path(root, k, metadata[k][i]['filename']))
            ds['image'].append(Path(root, k, metadata[k][i]['filename']))
            ds['image'].append(Path(root, k, metadata[k][i]['filename']))
            
            ds['sentence'].append(metadata[k][i]['raw'])
            ds['sentence'].append(metadata[k][i]['raw_1'])
            ds['sentence'].append(metadata[k][i]['raw_2'])
            ds['sentence'].append(metadata[k][i]['raw_3'])
            
            ds['identifier'].append(metadata[k][i]['filename'].replace(".jpg", f"_{i}"))
            ds['identifier'].append(metadata[k][i]['filename'].replace(".jpg", f"_{i}"))
            ds['identifier'].append(metadata[k][i]['filename'].replace(".jpg", f"_{i}"))
            ds['identifier'].append(metadata[k][i]['filename'].replace(".jpg", f"_{i}"))

    # Convert lists to numpy arrays
    for k in list(datasets.keys()):
        for split in list(datasets[k].keys()):
            datasets[k][split] = np.array(datasets[k][split])

    return datasets

class NWPU_Captions(torch.utils.data.Dataset):
    def __init__(self, root, annotations_file, split, transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']

        annotations = read_annotations_file(root, annotations_file)[split]

        self.images = annotations['image']
        self.sentences = annotations['sentence']
        self.identifier = annotations['identifier']

        self.mask = np.ones_like(self.images, dtype=bool)
        self.update_masked_arrays()

        self.transform = transform
        self.target_transform = target_transform

        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    def __len__(self):
        return len(self.masked_images)

    def max_length(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.masked_images[idx])
        sentence = self.masked_sentences[idx]
        identifier = self.masked_identifier[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            sentence = self.target_transform(sentence)

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        
        return pixel_values, sentence, identifier

    def update_masked_arrays(self):        
        # Update the masked arrays
        self.masked_images = self.images[self.mask]
        self.masked_sentences = self.sentences[self.mask]
        self.masked_identifier = self.identifier[self.mask]

    def random_labeling(self, elements):
        # Update the mask, set #elementes to true
        idx = np.argwhere(self.mask == False)
        np.random.shuffle(idx)
        idx = idx[:elements]
        self.mask[idx] = True

        self.update_masked_arrays()

    def set_empty_mask(self):
        self.mask = np.zeros_like(self.images, dtype=bool)
        self.update_masked_arrays()

if __name__ == '__main__':
    dataset = NWPU_Captions(root='../NWPU_images', annotations_file='../dataset_nwpu.json', split='test', target_transform=ToTensor(), mask=torch.zeros(100))
    print(len(dataset))
