import os.path
import torch
from tqdm import tqdm
import numpy as np
import sys
from torch.utils.data import DataLoader
from train_sentence_selectors.train_sentence_selector_LLM import load_llm_dataset
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.load_data import load_dataset
from blingfire import text_to_sentences
from auxiliary.split_texts import split_sample
from settings import data_dir
import json
from train_sentence_selectors.sentence_selection_data_and_eval import (evaluate_sentence_classifier, ImportanceSentenceDataset,
                                                                   collate_fn, load_LLM_dataset, load_importance_dataset, load_entropy_dataset)
import itertools
from settings import device, model_type


def create_importance_dataset(tokenizer, raw_dataset):
    if os.path.exists(data_dir("BERT_importance_sentence_assessment.json")):
        with open(data_dir("BERT_importance_sentence_assessment.json"), "r") as f:
            return json.load(f)

    classifiers = []
    for idx in range(3):
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
        model.load_state_dict(torch.load(f"saved_models/average_clf-{idx}.pt"))
        model.eval()
        classifiers.append(model)

    importance_dataset = {}
    with torch.no_grad():
        for split in raw_dataset:
            importance_dataset[split] = []
            for sample in tqdm(raw_dataset[split], file=sys.stdout, desc="Preparing importance dataset"):
                sentences = text_to_sentences(sample["text"]).split("\n")

                base_input = " ".join(sentences)
                word_lists, _, _ = split_sample(tokenizer, base_input, 500, None, 10)
                base_input = tokenizer(word_lists, max_length=510, truncation=True, padding=True, return_tensors="pt").to(device)

                scores = {i: [] for i in range(len(sentences))}

                for clf in classifiers:
                    base_prediction = clf(**base_input).logits.mean(0).cpu().numpy()

                    for leave_out_idx in range(len(sentences)):
                        input = " ".join(sentences[:leave_out_idx]) + " " + " ".join(sentences[leave_out_idx+1:])
                        word_lists, _, _ = split_sample(tokenizer, input, 500, None, 10)
                        input = tokenizer(word_lists, max_length=510, truncation=True, padding=True, return_tensors="pt").to(device)
                        prediction = clf(**input).logits.mean(0).cpu().numpy()
                        difference = np.abs(prediction - base_prediction).sum()
                        scores[leave_out_idx].append(difference)

                scores = np.array([np.mean(scores[i]) for i in range(len(sentences))])
                scores = scores - np.min(scores)
                importance_dataset[split].append([[float(x) for x in scores], sentences])
                with open(data_dir("BERT_importance_sentence_assessment.json"), "w") as f:
                    json.dump(importance_dataset, f)

    return importance_dataset

def train_sentence_classifier():
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    raw_dataset = load_dataset(tokenizer)
    importance_dataset = create_importance_dataset(tokenizer, raw_dataset)
    LLM_dataset = load_llm_dataset(raw_dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=3).to(device)

    train_dataset = ImportanceSentenceDataset(importance_dataset["train"], raw_dataset["train"], tokenizer)
    train_loader = iter(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)))
    train_batch_generator = itertools.cycle(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    best_score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"], importance_dataset=importance_dataset["val"])["BERT importance"]
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    epochs_without_improvement = 0
    loss_avg = None
    for epoch in range(200):
        model.train()

        for _ in (bar := tqdm(range(500), desc=f"Epoch {epoch}", file=sys.stdout)):
            inputs, labels = next(train_batch_generator)
            outputs = model(**inputs)

            loss = criterion(outputs.logits, labels.to(torch.long))
            scale_mask = (labels == 0)
            loss[scale_mask] *= 0.7
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            optimizer.step()
            optimizer.zero_grad()

            if loss_avg is None:
                loss_avg = loss.item()
            else:
                loss_avg = 0.95 * loss_avg + 0.05 * loss.item()
            bar.desc = f"Epoch {epoch}, Loss: {loss_avg:.2f}"

        score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"], importance_dataset=importance_dataset["val"])["BERT importance"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/sentence_classifier_importance.pt")
            epochs_without_improvement = 0
            print("New best! Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print("Early stopping triggered")
            break
        print()


def test_sentence_selection_model():
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    raw_dataset = load_dataset(tokenizer)
    LLM_dataset = load_LLM_dataset()
    importance_dataset = load_importance_dataset()
    entropy_dataset = load_entropy_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=3).to(device)
    model.load_state_dict(torch.load(f"saved_models/sentence_classifier_importance.pt"))

    evaluate_sentence_classifier(model, tokenizer, raw_dataset["test"], LLM_dataset["test"], importance_dataset=importance_dataset["test"], entropy_dataset=entropy_dataset["test"])

if __name__ == "__main__":
    #train_sentence_classifier()
    test_sentence_selection_model()