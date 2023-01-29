import torch
from model import ImageCaptioningSystem
from pathlib import Path
import os
from sklearn.cluster import KMeans
import numpy as np
import time


def load_embeddings(
    path: str,
    type: str,
    expected_num_files: int,
):
    if type == "image":
        return load_image_embeddings(path, expected_num_files)
    elif type == "text":
        return load_text_embeddings(path, expected_num_files)


def load_text_embeddings(
    path: str,
    expected_num_files: int,
):
    ...


def load_image_embeddings(
    path: str,
    expected_num_files: int,
):
    while True:
        embedding_files = os.listdir(path)
        embedding_files = [
            file for file in embedding_files if file.startswith("img_embeddings")
        ]

        id_files = os.listdir(path)
        id_files = [file for file in id_files if file.startswith("img_ids")]

        if len(embedding_files) == expected_num_files:
            break

        time.sleep(0.1)

    embeddings = torch.cat(
        [torch.load(Path(path, file)) for file in embedding_files], axis=0
    )

    embeddings = embeddings.flatten(start_dim=1)
    embeddings = embeddings.detach().cpu().numpy()

    ids = torch.cat([torch.load(Path(path, file)) for file in id_files], axis=0)
    ids = ids.detach().cpu().numpy()

    return embeddings, ids


def diversity_based_sample(
    path: str,
    num_clusters: int,
    type: str,
    expected_num_files: int,
):
    embeddings, ids = load_embeddings(path, type, expected_num_files)

    cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    label = cluster.labels_

    selected_ids = [np.random.choice(ids[label == i]) for i in range(num_clusters)]
    return torch.tensor(selected_ids)
