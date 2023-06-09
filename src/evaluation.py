from nltk import word_tokenize
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate import bleu_score
import numpy as np
from typing import List
import os
import torch
from pathlib import Path
from transformers import GPT2TokenizerFast
from rouge import Rouge


nltk.download("wordnet")
nltk.download("punkt")

tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
r = Rouge()


def rouge(references: List[str], hypothesis: str):
    return r.calc_score([hypothesis], references)


def meteor(references: List[str], hypothesis: str):
    splits_references = [reference.split() for reference in references]
    split_hypothesis = hypothesis.split()
    return meteor_score(
        splits_references,
        split_hypothesis,
    )


def bleu(references: List[str], hypothesis: str):
    splits_references = [reference.split() for reference in references]
    split_hypothesis = hypothesis.split()
    return bleu_score.sentence_bleu(splits_references, split_hypothesis)


def save_hypothesis(predction_path: str, hypothesis: List[str]):
    # save the hypothesis in a txt file
    file_path = Path(predction_path, "hypothesis.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for hypothesis_entry in hypothesis:
            file.write(hypothesis_entry)
            file.write("\n")


def load_references(references_path: List[str]):
    # load the references files given by the path and then split them line by line
    # result: [ [ref1, ref1, ...], [ref2, ref2, ...], ...]
    references = [open(path, "r").read().split("\n") for path in references_path]
    # transform them to a list of format: [ [ref1, ref2, ...], [ref1, ref2, ...], ...]
    # this trick is explained in https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
    references = list(zip(*references))

    return references


def load_hypothesis(prediction_path: str, hypothesis_file: str = "hypothesis.txt"):

    # load the ids and predicted tokens results from the different gpus(rank)
    # and concatenate them

    # get all files in the prediction path
    all_files = os.listdir(prediction_path)
    if hypothesis_file in all_files:
        hypothesis = (
            open(Path(prediction_path, hypothesis_file), "r").read().split("\n")
        )
        return hypothesis

    # filter them so only the ones img_ids and predicted_tokens are left
    id_files = [file for file in all_files if file.startswith("img_ids")]
    predicted_tokens_files = [
        file for file in all_files if file.startswith("predicted_tokens")
    ]

    # load the files and concatenate them
    ids = torch.cat(
        [torch.load(Path(prediction_path, file)) for file in id_files], axis=0
    )
    predicted_tokens = torch.cat(
        [torch.load(Path(prediction_path, file)) for file in predicted_tokens_files],
        axis=0,
    )

    # sort the tokens by the image id
    # this is done by getting the index of the img id like this
    # ids = [3,1,2] --[argsort]--> order = [2,0,1]
    # predicted_tokens = [4,5,6] --[order]--> predicted_tokens[order] = [5,6,4]
    order = ids.argsort()
    predicted_tokens = predicted_tokens[order]

    # decode the sorted tokens to sentences
    # result: [sentence1, sentence2, ...]
    hypothesis = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
    return hypothesis


def eval_validation(references_path: List[str], prediction_path: str):
    references = load_references(references_path)
    hypothesis = load_hypothesis(prediction_path)

    save_hypothesis(prediction_path, hypothesis)

    meteor_results = []
    bleu_results = []
    rouge_results = []
    for hypo, reference in zip(hypothesis, references):

        hypo = hypo.replace(".", "")
        reference = [ref.replace(" .", "") for ref in reference]

        bl = bleu(reference, hypo)
        me = meteor(reference, hypo)
        ro = rouge(reference, hypo)

        meteor_results.append(me)
        bleu_results.append(bl)
        rouge_results.append(ro)

    mean_bleu = np.mean(bleu_results)
    mean_meteor = np.mean(meteor_results)
    mean_rouge = np.mean(rouge_results)
    return mean_bleu, mean_meteor, mean_rouge
