import torch
import torchvision.datasets as datasets
import PIL
import pandas as pd
import json
import unicodedata
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
    
    for k in list(metadata.keys())[:20]:
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

    return datasets

class NWPU_Captions(torch.utils.data.Dataset):
    def __init__(self, root, annotations_file, split, transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.annotations = read_annotations_file(root, annotations_file)[split]
        self.transform = transform
        self.target_transform = target_transform

        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def __len__(self):
        return len(self.annotations['image'])

    def __getitem__(self, idx):
        image = PIL.Image.open(self.annotations['image'][idx])
        sentence = self.annotations['sentence'][idx]
        identifier = self.annotations['identifier'][idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            sentence = self.target_transform(sentence)

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        

        return pixel_values, sentence, identifier

if __name__ == '__main__':
    dataset = NWPU_Captions(root='NWPU_images', annotations_file='dataset_nwpu.json', split='test', target_transform=ToTensor())
    print(len(dataset))
