import torch
from model import ImageCaptioningSystem
from pathlib import Path
import os
from sklearn.cluster import KMeans
import numpy as np


def diversity_based_sample(
    path: str,
    num_clusters: int,
    type: str,
):
    if type == "image":
        # load image embeddings
        embedding_files = os.listdir(path)
        embedding_files = [
            file for file in embedding_files if file.startswith("img_embeddings")
        ]
        embeddings = torch.cat(
            [torch.load(Path(path, file)) for file in embedding_files], axis=0
        )

        embeddings = embeddings.flatten(start_dim=1)
        embeddings = embeddings.detach().cpu().numpy()

        id_files = os.listdir(path)
        id_files = [file for file in id_files if file.startswith("img_ids")]
        ids = torch.cat([torch.load(Path(path, file)) for file in id_files], axis=0)
        ids = ids.detach().cpu().numpy()

    label = cluster(embeddings, num_clusters)

    selected_ids = [np.random.choice(ids[label == i]) for i in range(num_clusters)]
    return torch.tensor(selected_ids)


def cluster(
    embeddings: np.ndarray,
    num_clusters: int,
):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_
