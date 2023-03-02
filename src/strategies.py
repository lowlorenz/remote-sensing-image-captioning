import torch
from model import ImageCaptioningSystem
from pathlib import Path
import os
from sklearn.cluster import KMeans
import numpy as np
import time
from transformers import GPT2TokenizerFast, BertTokenizer, BertModel
import math


def get_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    gpt_tokenizer = GPT2TokenizerFast.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    return bert, bert_tokenizer, gpt_tokenizer, device


def conf_and_cluster(
    path, elems_to_add, expected_num_files, embedding_type="image", mode="least", conf_average="sentence"
):
    # Select least confident data points
    least_conf_ids = confidence_sample(path, elems_to_add * 4, mode, conf_average)
    # Cluster
    returned_ids = diversity_based_sample(
        path=path,
        num_clusters=elems_to_add,
        embedding_type=embedding_type,
        expected_num_files=expected_num_files,
        conf_ids=least_conf_ids,
    )

    return torch.tensor(returned_ids)


def cluster_and_conf(
    path, elems_to_add, expected_num_files, embedding_type="image", mode="least", conf_average="sentence"
):
    # Cluster
    cluster_ids = diversity_based_sample(
        path=path,
        num_clusters=elems_to_add * 4,
        embedding_type=embedding_type,
        expected_num_files=expected_num_files,
    )
    # Select least confident data points
    returned_ids = confidence_sample(
        path=path,
        elems_to_add=elems_to_add,
        mode=mode,
        average=conf_average,
        cluster_ids=cluster_ids)

    return torch.tensor(returned_ids)


def load_embeddings(
    path: str,
    embedding_type: str,
    expected_num_files: int,
    conf_ids=None,
):
    if embedding_type == "image":
        embeddings, ids = load_image_embeddings(path, expected_num_files)
    elif embedding_type == "text":
        embeddings, ids = load_text_embeddings(path, expected_num_files)

    if conf_ids is not None:
        mask = np.isin(ids, conf_ids.numpy())
        ids = ids[mask]
        embeddings = embeddings[mask]

    return embeddings, ids


def load_text_embeddings(
    path: str,
    expected_num_files: int,
):
    while True:
        predicted_tokens_files = os.listdir(path)
        predicted_tokens_files = [
            file
            for file in predicted_tokens_files
            if file.startswith("predicted_tokens")
        ]

        id_files = os.listdir(path)
        id_files = [file for file in id_files if file.startswith("img_ids")]

        if len(id_files) == expected_num_files:
            break

        time.sleep(0.1)

    predicted_tokens = torch.cat(
        [torch.load(Path(path, file)) for file in predicted_tokens_files], axis=0
    )

    bert, bert_tokenizer, gpt_tokenizer, device = get_tokenizer()

    hypothesis = gpt_tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
    with torch.no_grad():
        all_tokens = bert_tokenizer(hypothesis, padding="longest").input_ids
        all_tokens = torch.tensor(all_tokens)
        num_batches = math.ceil(all_tokens.shape[0] / 32)
        embeddings = []
        for i in range(num_batches):
            tokens = all_tokens[i * 32 : (i + 1) * 32].to(device)
            e = bert(tokens).pooler_output.detach().cpu()
            embeddings.append(e)

    embeddings = torch.cat(embeddings, axis=0).numpy()

    ids = torch.cat([torch.load(Path(path, file)) for file in id_files], axis=0)
    ids = ids.detach().cpu().numpy()

    return embeddings, ids


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


def confidence_sample(path, elems_to_add, mode="least", average="sentence", cluster_ids=None):
    # load image ids
    id_files = os.listdir(path)
    id_files = [f for f in id_files if f.startswith("img_ids")]
    ids = torch.cat([torch.load(Path(path, f)) for f in id_files], axis=0)
    ids = ids.flatten()
    ids = ids.detach().cpu()

    # if cluster_ids is not None, then only keep the ids that are in the cluster_ids
    if cluster_ids is not None:
        mask = np.isin(ids, cluster_ids.numpy())
    else:
        mask = np.full(ids.shape, True)

    confidence_files = os.listdir(path)

    # when least confidence is selected, only load the confidence files
    if mode == "least" and average == 'sentence':
        confidence_files = [f for f in confidence_files if f.startswith("confidence")]
    elif mode == "least" and average == 'word':
        confidence_files = [f for f in confidence_files if f.startswith("word_confidence")]

    # when margin confidence is selected, only load the margin files
    elif mode == "margin" and average == "sentence":
        confidence_files = [f for f in confidence_files if f.startswith("margin")]
    elif mode == "margin" and average == "word":
        confidence_files = [f for f in confidence_files if f.startswith("word_margin")]

    # concatenate all the confidence files
    confidences = torch.cat(
        [torch.load(Path(path, f)) for f in confidence_files], axis=0
    )[torch.tensor(mask)]

    # flatten the tensor and convert it to a list, so that it can be zipped and sorted
    confidences = confidences.flatten()
    confidences = list(confidences.detach().cpu())

    # sort the ids based on the confidence
    joint_list = [a for a in zip(confidences, ids)]
    joint_list.sort(key=lambda l: l[0])

    # keep the img_ids of the highest confidence values and drop the rest
    returned_ids = [ident[1] for ident in joint_list[:elems_to_add]]

    return torch.tensor(returned_ids)


def diversity_based_sample(
    path: str,
    num_clusters: int,
    embedding_type: str,
    expected_num_files: int,
    conf_ids: torch.tensor = None,
):
    embeddings, ids = load_embeddings(path, embedding_type, expected_num_files, conf_ids)

    cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    label = cluster.labels_

    selected_ids = []
    for i in range(num_clusters):
        cluster_members = ids[label == i]

        if len(cluster_members) == 0:
            print(
                "Cluster is empty, selecting no entry",
                embeddings.shape,
                ids.shape,
                label.shape,
                num_clusters,
            )
            continue

        selected_ids.append(np.random.choice(cluster_members))
    return torch.tensor(selected_ids)
