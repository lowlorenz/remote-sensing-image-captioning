import torch
import torchvision.datasets as datasets
import PIL
import pandas as pd
import json
import unicodedata
import numpy as np
from pathlib import Path
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoProcessor,
)
import random


def read_annotations_file(root, annotation_file):
    file_content = open(annotation_file, encoding="utf8").read()
    file_content = unicodedata.normalize("NFKD", file_content)
    metadata = json.loads(file_content)

    datasets = {
        "train": {"image": [], "img_id": [], "sentences": [], "sentences_id": []},
        "val": {"image": [], "img_id": [], "sentences": [], "sentences_id": []},
        "test": {"image": [], "img_id": [], "sentences": [], "sentences_id": []},
    }

    # Scraping the data from the json file
    # 1. loop over the keys of the json file
    for key in list(metadata.keys()):
        # 2. loop over the images
        for i in range(len(metadata[key])):
            # retrieve the data split from the json
            split = metadata[key][i]["split"]

            # select the dataset to append the data based on the split
            ds = datasets[split]

            ds["image"].append(Path(root, key, metadata[key][i]["filename"]))
            ds["img_id"].append(metadata[key][i]["imgid"])

            sentence_keys = ["raw", "raw_1", "raw_2", "raw_3", "raw_4"]
            sentences = [
                metadata[key][i][sentence_key] for sentence_key in sentence_keys
            ]
            id_keys = ["sentids", "sentids_1", "sentids_2", "sentids_3", "sentids_4"]
            ids = [metadata[key][i][id_key] for id_key in id_keys]

            ds["sentences"].append(sentences)
            ds["sentences_id"].append(ids)

    return datasets


class NWPU_Captions(torch.utils.data.Dataset):
    def __init__(self, cfg, split, seed=42):
        assert split in ["train", "val", "test"]

        images_path = Path(cfg.dataset.data_path, "NWPU_images")
        annotations_path = Path(cfg.dataset.data_path, "dataset_nwpu.json")

        annotations = read_annotations_file(images_path, annotations_path)[split]

        self.images = annotations["image"]
        self.sentences = annotations["sentences"]
        self.seed = seed

        if "processor_path" in cfg.model.keys():
            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            self.tokenizer = processor.tokenizer
            self.feature_extractor = processor.image_processor
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.tokenizer_path)



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx]).convert("RGB")
        index = random.randint(0, 4)
        sentence = self.sentences[idx][index]

        # if self.model.name == "gpt2":
        #     image = torchvision.tra(image)

        img_features = self.feature_extractor(
            image,
            return_tensors="pt",
        )['pixel_values'].squeeze()

        sentence_features = self.tokenizer(
            sentence,
            return_tensors="pt",
            max_length=120,
            padding="max_length",
        )['input_ids'][0]

        return (
            img_features,
            sentence_features
        )


if __name__ == "__main__":
    pass
    # from model import ImageCaptioningSystem

    # dataset = NWPU_Captions(
    #     root="NWPU-Captions/NWPU_images",
    #     annotations_file="NWPU-Captions/dataset_nwpu.json",
    #     split="test",
    #     transform=ToTensor(),
    # )
    # print(len(dataset))

    # test_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=False, num_workers=4
    # )
    # it = iter(test_loader)
    # batch = next(it)

    # model = ImageCaptioningSystem(
    #     lr=0.001,
    #     mutliple_sentence_loss=False,
    # )

    # model.training_step(batch, 0)
