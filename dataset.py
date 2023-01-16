import torch
import torchvision.datasets as datasets
import PIL
import pandas as pd
import json
import unicodedata
import numpy as np
from pathlib import Path
from transformers import ViTFeatureExtractor, GPT2TokenizerFast
from torchvision.transforms import ToTensor


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


def vectorize_annotations(dataset):
    # Tokenize the sentences
    dataset["sentences"] = tokenize_sentences(dataset["sentences"])
    # Convert lists to arrays
    dataset["image"] = np.array(dataset["image"])
    dataset["img_id"] = np.array(dataset["img_id"])
    dataset["sentences_id"] = np.array(dataset["sentences_id"])
    return dataset


def tokenize_sentences(sentences):
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    # Convert the sentences to token-tensors by looping over the
    # sentences list and calling the tokenizer on every entry
    sentence_tensor = np.zeros((len(sentences), 5, 56), dtype=int)
    for i in range(len(sentences)):
        for j in range(5):

            sentence_tensor[i, j] = tokenizer(
                sentences[i][j],
                max_length=56,
                padding="max_length",
                return_tensors="np",
            ).input_ids
    return sentence_tensor


class NWPU_Captions(torch.utils.data.Dataset):
    def __init__(self, root, annotations_file, split, transform=None):
        assert split in ["train", "val", "test"]

        annotations = read_annotations_file(root, annotations_file)[split]
        annotations = vectorize_annotations(annotations)

        self.images = annotations["image"]
        self.sentences = annotations["sentences"]
        self.img_ids = annotations["img_id"]
        self.sentences_ids = annotations["sentences_id"]

        self.mask = np.ones_like(self.images, dtype=bool)
        self.update_masked_arrays()

        self.transform = transform

        self.image_processor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

    def __len__(self):
        return len(self.masked_images)

    def max_length(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.masked_images[idx])

        if self.transform:
            image = self.transform(image)

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        sentences = self.masked_sentences[idx]
        img_id = self.masked_img_ids[idx]
        sentences_ids = self.masked_sentences_ids[idx]

        return pixel_values, sentences, img_id, sentences_ids

    def update_masked_arrays(self):
        # Update the masked arrays
        self.masked_images = self.images[self.mask]
        self.masked_sentences = self.sentences[self.mask]
        self.masked_img_ids = self.img_ids[self.mask]
        self.masked_sentences_ids = self.sentences_ids[self.mask]

    def add_random_labels(self, num_elements):
        # Update the mask, set num elementes to true
        idx = np.argwhere(self.mask == False)
        np.random.shuffle(idx)
        idx = idx[:num_elements]
        self.mask[idx] = True

        self.update_masked_arrays()

    def set_empty_mask(self):
        self.mask = np.zeros_like(self.images, dtype=bool)
        self.update_masked_arrays()


if __name__ == "__main__":
    from model import ImageCaptioningSystem

    dataset = NWPU_Captions(
        root="NWPU-Captions/NWPU_images",
        annotations_file="NWPU-Captions/dataset_nwpu.json",
        split="test",
        transform=ToTensor(),
    )
    print(len(dataset))

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4
    )
    it = iter(test_loader)
    batch = next(it)

    model = ImageCaptioningSystem(0.01)

    model.training_step(batch, 0)
    # pixel_values, sentences, img_id, sentences_ids = batch
