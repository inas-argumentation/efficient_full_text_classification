import random
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import torch
from blingfire import text_to_sentences
from sklearn.metrics import ndcg_score
from settings import data_dir
import json
from settings import device

def load_entropy_dataset():
    try:
        with open(data_dir("BERT_entropy_dataset.json"), "r") as f:
            return json.load(f)
    except:
        print("Entropy dataset must be created first.")
        quit()


def load_importance_dataset():
    try:
        with open(data_dir("BERT_importance_sentence_assessment.json"), "r") as f:
            return json.load(f)
    except:
        print("Importance dataset must be created first.")
        quit()

def load_LLM_dataset():
    try:
        with open(data_dir("LLM_sentence_assessment_new.json"), "r") as f:
            return json.load(f)
    except:
        print("LLM dataset must be created first.")
        quit()

def collate_fn(batch, tokenizer, max_length=512):
    sentences = [item['sentence'] for item in batch]
    labels = [item['label'] for item in batch]

    inputs = tokenizer(sentences, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to("cuda")
    labels = torch.tensor(labels, dtype=torch.float, device="cuda")

    return inputs, labels

def create_input_evidence(tokenizer, sample, sentence_idx):
    sentences = sample["evidence_sentence_annotation"]
    sentence_tokens = len(tokenizer.tokenize(sentences[sentence_idx][2]))

    add_sentences_before = [x[2] for x in sentences[max(sentence_idx-3, 0):sentence_idx]]
    while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
        add_sentences_before.pop(0)

    input_text = (sample["species"] + f"; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentences[sentence_idx][2] +
          f" {tokenizer.sep_token} " + " ".join([x[2] for x in sentences[sentence_idx+1:min(len(sentences), sentence_idx+4)]]))

    return input_text

class EvidenceSentenceDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.sentences = []
        self.labels = []

        for sample in tqdm(samples, file=sys.stdout, desc="Creating evidence dataset"):
            if sample["evidence_array"] is not None:
                self.sentences += [create_input_evidence(tokenizer, sample, i) for i in range(len(sample["evidence_sentence_annotation"]))]
                self.labels += [x[3] for x in sample["evidence_sentence_annotation"]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label': self.labels[idx]
        }

def create_input_LLM(sentence, doc_sentences, species, tokenizer):
    correct_idx = doc_sentences.index(sentence)
    sentence_tokens = len(tokenizer.tokenize(sentence))

    add_sentences_before = doc_sentences[max(correct_idx - 3, 0):correct_idx]
    while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
        add_sentences_before.pop(0)

    input_text = (species + f"; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentence + f" {tokenizer.sep_token} " + " ".join(
                doc_sentences[correct_idx + 1:min(len(doc_sentences), correct_idx + 4)]))

    return input_text

class LLMSentenceDataset(Dataset):
    def __init__(self, LLM_assessments, corpus, tokenizer):
        self.sentences = []
        self.labels = []

        for entry in tqdm(LLM_assessments, file=sys.stdout, desc="Creating LLM dataset"):
            doc = [x for x in corpus if x["dataset_idx"] == entry["dataset_idx"]][0]
            doc_sentences = text_to_sentences(doc["text"]).split("\n")
            for assessment in entry["LLM_assessments"]:
                self.sentences.append(create_input_LLM(assessment[0], doc_sentences, doc["species"], tokenizer))
                self.labels.append(assessment[1])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label': self.labels[idx]
        }

class ImportanceSentenceDataset(Dataset):
    def __init__(self, importance_dataset, corpus, tokenizer):
        self.sentences = []
        self.labels = []

        for entry in tqdm(importance_dataset, file=sys.stdout, desc="Creating importance dataset"):
            doc = [x for x in corpus if entry[1][0] in x["text"] and entry[1][-1] in x["text"]][0]
            lower_threshold = np.percentile(entry[0], 50.0)
            upper_threshold = np.percentile(entry[0], 80.0)
            for sentence_idx in range(len(entry[1])):
                self.sentences.append(create_input_importance(tokenizer, doc, entry, sentence_idx))
                self.labels.append(0 if entry[0][sentence_idx] < lower_threshold else (1 if entry[0][sentence_idx] < upper_threshold else 2))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label': self.labels[idx]
        }

def create_input_importance(tokenizer, sample, importance_data, sentence_idx):
    all_sentences = importance_data[1]
    sentence = all_sentences[sentence_idx]
    sentence_tokens = len(tokenizer.tokenize(sentence))

    add_sentences_before = all_sentences[max(sentence_idx-3, 0):sentence_idx]
    while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
        add_sentences_before.pop(0)

    input_text = (sample["species"] + "; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentence +
          f" {tokenizer.sep_token} " + " ".join(all_sentences[sentence_idx+1:min(len(all_sentences), sentence_idx+4)]))
    return input_text

def calculate_LLM_sample_ndcg(model, tokenizer, sample, LLM_assessments, batch_size=8):
    doc_sentences = text_to_sentences(sample["text"]).split("\n")
    inputs = []
    labels = []
    for assessment in LLM_assessments["LLM_assessments"]:
        input_text = create_input_LLM(assessment[0], doc_sentences, sample["species"], tokenizer)
        inputs.append(input_text)
        labels.append(assessment[1])

    predictions = []
    for idx in range(0, len(inputs), batch_size):
        batch_inputs = tokenizer(inputs[idx:idx + batch_size], return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")

        output = model(**batch_inputs).logits
        if output.shape[-1] == 3:
            output = (torch.softmax(output, dim=-1) * torch.tensor([[0, 1, 2]], device=device, dtype=torch.float)).sum(-1).detach().cpu().numpy()
        else:
            output = torch.sigmoid(output).squeeze(-1).detach().cpu().numpy()
        predictions += output.tolist()

    labels = np.array(labels).astype("float")
    ndcg = ndcg_score(labels.reshape(1, -1), np.array(predictions).reshape(1, -1))
    return ndcg

def calculate_evidence_sample_ndcg(model, tokenizer, sample, batch_size=8):
    sentences = sample["evidence_sentence_annotation"]

    texts = [create_input_evidence(tokenizer, sample, i) for i in range(len(sentences))]
    labels = [s[3] for s in sentences]

    predictions = []
    for idx in range(0, len(texts), batch_size):
        batch_inputs = tokenizer(texts[idx:idx + batch_size], return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")

        output = model(**batch_inputs).logits
        if output.shape[-1] == 3:
            output = (torch.softmax(output, dim=-1) * torch.tensor([[0, 1, 2]], device=device, dtype=torch.float)).sum(-1).detach().cpu().numpy()
        else:
            output = torch.sigmoid(output).squeeze(-1).detach().cpu().numpy()
        predictions += output.tolist()

    labels = np.array(labels).astype("float")
    ndcg = ndcg_score(labels.reshape(1, -1), np.array(predictions).reshape(1, -1))
    return ndcg

def calculate_importance_sample_ndcg(model, tokenizer, sample, importance_data, batch_size=8):
    texts = [create_input_importance(tokenizer, sample, importance_data, i) for i in range(len(importance_data[1]))]
    scores = importance_data[0]

    predictions = []
    for idx in range(0, len(texts), batch_size):
        batch_inputs = tokenizer(texts[idx:idx + batch_size], return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")

        output = model(**batch_inputs).logits
        if output.shape[-1] == 3:
            output = (torch.softmax(output, dim=-1) * torch.tensor([[0, 1, 2]], device=device, dtype=torch.float)).sum(-1).detach().cpu().numpy()
        else:
            output = output.squeeze(-1).detach().cpu().numpy()
        predictions += output.tolist()

    scores = np.array(scores).astype("float")
    ndcg = ndcg_score(scores.reshape(1, -1), np.array(predictions).reshape(1, -1))
    return ndcg

def evaluate_sentence_classifier(model, tokenizer, raw_dataset, LLM_dataset, importance_dataset=None, entropy_dataset=None):
    model.eval()
    result = {}

    LLM_ndcg_scores = []
    with torch.no_grad():
        for LLM_sample in tqdm(LLM_dataset, desc=f"Evaluating LLM ndcg", file=sys.stdout):
            dataset_sample = [x for x in raw_dataset if x["dataset_idx"] == LLM_sample["dataset_idx"]][0]
            LLM_ndcg_scores.append(calculate_LLM_sample_ndcg(model, tokenizer, dataset_sample, LLM_sample))
    avg_LLM_ndcg_score = np.mean(LLM_ndcg_scores)
    result["LLM"] = avg_LLM_ndcg_score

    evidence_ndcg_scores = []
    with torch.no_grad():
        for sample in tqdm(raw_dataset, desc=f"Evaluating evidence ndcg", file=sys.stdout):
            if sample["evidence_array"] is not None:
                evidence_ndcg_scores.append(calculate_evidence_sample_ndcg(model, tokenizer, sample))
    avg_evidence_ndcg_score = np.mean(evidence_ndcg_scores)
    result["Evidence"] = avg_evidence_ndcg_score

    if importance_dataset is not None:
        BERT_importance_ndcg_scores = []
        with torch.no_grad():
            for sample in tqdm(importance_dataset, desc=f"Evaluating BERT importance ndcg", file=sys.stdout):
                dataset_sample = [x for x in raw_dataset if sample[1][0] in x["text"] and sample[1][-1] in x["text"]][0]
                BERT_importance_ndcg_scores.append(calculate_importance_sample_ndcg(model, tokenizer, dataset_sample, sample))

        avg_BERT_importance_ndcg_score = np.mean(BERT_importance_ndcg_scores)
        result["BERT importance"] = avg_BERT_importance_ndcg_score

    if entropy_dataset is not None:
        entropy_ndcg_scores = []
        with torch.no_grad():
            for sample in tqdm(entropy_dataset, desc=f"Evaluating entropy ndcg", file=sys.stdout):
                dataset_sample = [x for x in raw_dataset if sample[1][0] in x["text"] and sample[1][-1] in x["text"]][0]
                entropy_ndcg_scores.append(calculate_importance_sample_ndcg(model, tokenizer, dataset_sample, sample))

        avg_entropy_ndcg_score = np.mean(entropy_ndcg_scores)
        result["Entropy"] = avg_entropy_ndcg_score

    for score in result:
        print(f"{score} NDCG score: {result[score]:.3f}")
    return result