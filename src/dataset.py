import PIL
import json
import numpy as np
from pathlib import Path
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoProcessor,
)
import torch.multiprocessing

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from abc import ABC, abstractmethod

torch.multiprocessing.set_sharing_strategy('file_system')

class BaseDataset(Dataset, ABC):
    def __init__(self, cfg, split, transform=None):

        if "processor_path" in cfg.model.keys():
            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            self.tokenizer = processor.tokenizer
            self.feature_extractor = processor.image_processor
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                cfg.model.tokenizer_path
            )
            
        self.transform = transform
        self.data = []
        self.load_data(split, cfg.dataset.image_path, cfg.dataset.annotations_path)

    @abstractmethod
    def load_data(self, split, images_path, annotations_path):
        """
        Method to load data specific to each subclass.
        This is abstract and must be implemented by all subclasses.
        """
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        image = PIL.Image.open(entry["filename"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        img_features = self.feature_extractor(
            image,
            return_tensors="pt",
        )["pixel_values"].squeeze()

        tokenizer_output = self.tokenizer(
            entry["sentences"],
            return_tensors="pt",
            max_length=120,
            padding="max_length",
        )
        sentence_features = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]
        return (img_features, sentence_features, attention_mask)


class NWPU_Captions(BaseDataset):
    def load_data(self, split, images_path, annotations_path):
        with open(annotations_path, "r") as f:
            data_dict = json.load(f)

        self.data = []

        for category, entries in data_dict.items():
            for entry in entries:
                if entry["split"] == split:
                    # Collect all 'raw' sentences
                    sentences = [
                        entry[f"raw_{i}"] if f"raw_{i}" in entry else entry["raw"]
                        for i in range(5)
                    ]
                    self.data.append(
                        {
                            "filename": Path(images_path, category, entry["filename"]),
                            "sentences": sentences,
                        }
                    )


class RSICD(BaseDataset):
    def load_data(self, split, images_path, annotations_path):
        with open(annotations_path, "r") as f:
            data_dict = json.load(f)

        self.data = []

        for entry in data_dict["images"]:
            if entry["split"] == split:
                sentences = [sentence["raw"] for sentence in entry["sentences"]]
                self.data.append(
                    {
                        "filename": Path(images_path, entry["filename"]),
                        "sentences": sentences,
                    }
                )


Sydney_Captions = RSICD
UCM_Captions = RSICD
