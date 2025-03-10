import itertools
import os.path
import sys
import torch
from tqdm import tqdm
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from data.load_data import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from train_sentence_selectors.sentence_selection_data_and_eval import (EvidenceSentenceDataset, collate_fn,
                                                                       evaluate_sentence_classifier, load_importance_dataset, load_LLM_dataset, load_entropy_dataset)
from settings import model_type, device

def train_sentence_classifier():
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    raw_dataset = load_dataset(tokenizer)
    LLM_dataset = load_LLM_dataset()
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=1).to(device)

    train_dataset = EvidenceSentenceDataset(raw_dataset["train"], tokenizer)
    train_loader = iter(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)))
    train_batch_generator = itertools.cycle(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    best_score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"])["Evidence"]

    epochs_without_improvement = 0
    loss_avg = None
    for epoch in range(200):
        model.train()

        for _ in (bar := tqdm(range(500), desc=f"Epoch {epoch}", file=sys.stdout)):
            inputs, labels = next(train_batch_generator)
            outputs = model(**inputs)

            loss = criterion(outputs.logits.squeeze(-1), labels)
            loss = (loss * labels + loss * (1 - labels) * 0.15).mean()

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            optimizer.step()
            optimizer.zero_grad()

            if loss_avg is None:
                loss_avg = loss.item()
            else:
                loss_avg = 0.95 * loss_avg + 0.05 * loss.item()
            bar.desc = f"Epoch {epoch}, Loss: {loss_avg:.2f}"

        score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"])["Evidence"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/sentence_classifier_evidence.pt")
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

    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=1).to(device)
    model.load_state_dict(torch.load(f"saved_models/sentence_classifier_evidence.pt"))

    evaluate_sentence_classifier(model, tokenizer, raw_dataset["test"], LLM_dataset["test"], importance_dataset=importance_dataset["test"], entropy_dataset=entropy_dataset["test"])


if __name__ == "__main__":
    #train_classifier()
    test_classifier()