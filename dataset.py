import torch
import torchvision.datasets as datasets
import PIL
import pandas as pd
import json
import unicodedata
import numpy as np
from pathlib import Path
from transformers import ViTFeatureExtractor
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
    for key in list(metadata.keys()):
        for i in range(len(metadata[key])):
            ds = datasets[metadata[key][i]['split']]
            for idx in range(5):
                
                ds['image'].append(Path(root, key, metadata[key][i]['filename']))
                
                if idx == 0:
                    ds['sentence'].append(metadata[key][i]['raw'])
                else:
                    ds['sentence'].append(metadata[key][i][f'raw_{idx}'])
                
                ds['identifier'].append(metadata[key][i]['filename'].replace(".jpg", f"_{idx}"))

    # Convert lists to numpy arrays
    for split in list(datasets.keys()):
        datasets[split]['sentence'] = np.array(datasets[split]['sentence'], dtype=object)
        datasets[split]['image'] = np.array(datasets[split]['image'])
        datasets[split]['identifier'] = np.array(datasets[split]['identifier'])

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

        self.image_processor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
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

        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        
        return pixel_values, sentence, identifier

    def update_masked_arrays(self):        
        # Update the masked arrays
        self.masked_images = self.images[self.mask]
        self.masked_sentences = self.sentences[self.mask]
        self.masked_identifier = self.identifier[self.mask]

    def add_random_labels(self, elements):
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
    dataset = NWPU_Captions(root='NWPU_images', annotations_file='dataset_nwpu.json', split='test', target_transform=ToTensor(), mask=torch.zeros(100))
    print(len(dataset))
