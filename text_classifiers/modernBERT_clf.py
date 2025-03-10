import os.path
import torch
from tqdm import tqdm
import random
import numpy as np
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.load_data import load_dataset
from auxiliary.loss_fn import categorical_cross_entropy_with_logits
from auxiliary.evaluate_predictions import evaluate_classification_predictions
from blingfire import text_to_sentences
from collections import Counter
from settings import device

model_type = "answerdotai/ModernBERT-base"

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def create_one_hot_label(labels, num_classes=6):
    """Convert label list into one-hot encoded tensor."""
    one_hot = torch.zeros(num_classes)
    for label in labels:
        one_hot[label] = 1.0
    return one_hot

def create_text(tokenizer, text, randomized):
    if not randomized:
        return text

    sentences = text_to_sentences(text).split("\n")
    random_selection = list(sorted(list(range(len(sentences)))[:int(0.8*len(sentences))]))
    sentences = [sentences[x] for x in random_selection]

    while (current_length := len(tokenizer.tokenize(" ".join(sentences)))) > 8192:
        n_sentences_to_remove = max(1, int((1 - 8192 / current_length) * len(sentences)))
        for _ in range(n_sentences_to_remove):
            random_index = random.randrange(len(sentences))
            sentences.pop(random_index)
    return " ".join(sentences)

def evaluate(model, tokenizer, dataset, randomization, split):
    model.eval()
    all_predictions = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for sample in tqdm(dataset[split], desc=f"Evaluating {split}"):
            label = create_one_hot_label(sample['labels'])

            sampled_predictions = []
            sampled_losses = []
            for pred_idx in range(10 if randomization else 1):
                full_text = create_text(tokenizer, sample['text'], randomization)
                inputs = tokenizer([full_text], max_length=8192, truncation=True, padding=True, return_tensors="pt").to(device)

                output = model(**inputs).logits[0]
                prediction = torch.argmax(output, dim=-1).item()
                sampled_predictions.append(prediction)
                sampled_losses.append(categorical_cross_entropy_with_logits(output, label.to("cuda")).item())

            prediction = most_common(sampled_predictions)
            all_predictions.append(prediction)
            all_labels.append(label.numpy())
            all_losses.append(np.mean(sampled_losses))

    macro_f1, micro_f1 = evaluate_classification_predictions(np.array(all_predictions), np.array(all_labels), convert_predictions=True)
    avg_loss = np.mean(all_losses)
    print(f"{split} loss: {avg_loss:.3f}")
    return avg_loss, macro_f1, micro_f1

def train_classifier(save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    dataset = load_dataset(bert_tokenizer)

    label_counts = torch.stack([create_one_hot_label(x["labels"]) for x in dataset["train"]], dim=0).sum(0)
    class_weights = torch.tensor((1 / torch.sqrt(label_counts)) / (1 / torch.sqrt(label_counts)).mean(), device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    epochs_without_improvement = 0
    batch_size = 2
    n_grad_acc_steps = 8

    score_func = lambda x: (x[1] + x[2] - x[0])
    best_score = score_func(evaluate(model, tokenizer, dataset, "random" in save_name_end, 'val'))
    print(f"Validation Loss: {best_score:.4f}")

    for epoch in range(200):
        model.train()
        total_loss = 0
        random.shuffle(dataset['train'])

        batch_texts = []
        batch_labels = []
        for sample_idx, sample in enumerate(tqdm(dataset['train'], desc=f"Epoch {epoch}")):
            full_text = create_text(tokenizer, sample['text'], "random" in save_name_end)
            label = create_one_hot_label(sample['labels']).to(device)

            batch_texts.append(full_text)
            batch_labels.append(label)

            if len(batch_texts) >= batch_size:
                inputs = tokenizer(batch_texts, max_length=8192, truncation=True, padding=True, return_tensors="pt").to(device)
                outputs = model(**inputs).logits
                loss = (1/n_grad_acc_steps) * categorical_cross_entropy_with_logits(outputs, torch.stack(batch_labels, dim=0), class_weights.unsqueeze(0)).mean()

                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

                if (sample_idx+1) % n_grad_acc_steps == 0 or sample_idx == len(dataset["train"])-1:
                    optimizer.step()
                    optimizer.zero_grad()

                batch_texts = []
                batch_labels = []

        avg_loss = total_loss / len(dataset['train'])
        print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")

        score = score_func(evaluate(model, tokenizer, dataset, "random" in save_name_end, 'val'))
        print(f"Validation Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/modernBERT_clf{save_name_end}.pt")
            epochs_without_improvement = 0
            print("New best! Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print("Early stopping triggered")
            break

def test_classifier(save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    model.load_state_dict(torch.load(f"saved_models/modernBERT_clf{save_name_end}.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    dataset = load_dataset(bert_tokenizer)
    return evaluate(model, tokenizer, dataset, "random" in save_name_end, "test")

def perform_experiment(random_selection=False):
    for i in range(7):
        if not os.path.exists(f"saved_models/modernBERT_clf-{i}{'-random' if random_selection else ''}.pt"):
            train_classifier(f"-{i}{'-random' if random_selection else ''}")

    macro_scores = []
    micro_scores = []
    for i in range(7):
        _, macro_f1, micro_f1 = test_classifier(f"-{i}{'-random' if random_selection else ''}")
        macro_scores.append(macro_f1)
        micro_scores.append(micro_f1)

    print(f"Average Macro F1: {np.mean(macro_scores):.3f} ({[f'{x:.3f}' for x in macro_scores]})")
    print(f"Average Micro F1: {np.mean(micro_scores):.3f} ({[f'{x:.3f}' for x in micro_scores]})")

if __name__ == "__main__":
    perform_experiment(True)