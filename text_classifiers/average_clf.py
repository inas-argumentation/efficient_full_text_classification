import os.path
import torch
from tqdm import tqdm
import random
import numpy as np
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.load_data import load_dataset, load_paper_full_text
from auxiliary.split_texts import split_sample
from auxiliary.loss_fn import categorical_cross_entropy_with_logits
from auxiliary.evaluate_predictions import evaluate_classification_predictions
from settings import model_type, device

def create_one_hot_label(labels, num_classes=6):
    one_hot = torch.zeros(num_classes)
    for label in labels:
        one_hot[label] = 1.0
    return one_hot

def evaluate(model, tokenizer, dataset, split):
    model.eval()
    all_predictions = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for sample in tqdm(dataset[split], desc=f"Evaluating {split}"):
            full_text = sample['text']
            label = create_one_hot_label(sample['labels'])
            word_lists, _, _ = split_sample(tokenizer, full_text, 500, None, 50)
            inputs = tokenizer(word_lists, max_length=500, truncation=True, padding=True, return_tensors="pt").to(device)

            outputs = model(**inputs)
            avg_logits = torch.mean(outputs.logits, dim=0)
            prediction = torch.argmax(avg_logits, dim=-1).item()

            all_predictions.append(prediction)
            all_labels.append(label.numpy())
            all_losses.append(categorical_cross_entropy_with_logits(avg_logits, label.to(device)).item())

    macro_f1, micro_f1 = evaluate_classification_predictions(np.array(all_predictions), np.array(all_labels), convert_predictions=True)
    avg_loss = np.mean(all_losses)
    print(f"{split} loss: {avg_loss:.3f}")
    return avg_loss, macro_f1, micro_f1

def train_classifier(save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    dataset = load_dataset(tokenizer)

    label_counts = torch.stack([create_one_hot_label(x["labels"]) for x in dataset["train"]], dim=0).sum(0)
    class_weights = torch.tensor((1 / torch.sqrt(label_counts)) / (1 / torch.sqrt(label_counts)).mean(), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    epochs_without_improvement = 0
    batch_size = 32

    score_func = lambda x: (x[1] + x[2] - x[0])
    best_score = score_func(evaluate(model, tokenizer, dataset, 'val'))
    print(f"Validation Loss: {best_score:.4f}")

    for epoch in range(200):
        model.train()
        total_loss = 0
        random.shuffle(dataset['train'])

        batch_texts = []
        batch_labels = []
        batch_splits = []
        for sample_idx, sample in enumerate(tqdm(dataset['train'], desc=f"Epoch {epoch}")):
            full_text = sample['text']
            word_lists, arrays, _ = split_sample(tokenizer, full_text, 500, sample["evidence_array"], 50)
            label = create_one_hot_label(sample['labels']).to(device)

            batch_texts += word_lists
            batch_labels.append(label)
            batch_splits.append(len(word_lists))

            if len(batch_texts) >= batch_size:
                inputs = tokenizer(batch_texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
                outputs = model(**inputs).logits

                averaged_outputs = []
                current_idx = 0
                for n_splits in batch_splits:
                    averaged_outputs.append(outputs[current_idx:current_idx + n_splits].mean(0))
                    current_idx += n_splits
                averaged_outputs = torch.stack(averaged_outputs, dim=0)

                loss = categorical_cross_entropy_with_logits(averaged_outputs, torch.stack(batch_labels, dim=0), class_weights.unsqueeze(0)).mean()

                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

                optimizer.step()
                optimizer.zero_grad()

                batch_texts = []
                batch_labels = []
                batch_splits = []

        avg_loss = total_loss / len(dataset['train'])
        print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")

        score = score_func(evaluate(model, tokenizer, dataset, 'val'))
        print(f"Validation Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/average_clf{save_name_end}.pt")
            epochs_without_improvement = 0
            print("New best! Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print("Early stopping triggered")
            break

def test_classifier(save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    model.load_state_dict(torch.load(f"saved_models/average_clf{save_name_end}.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    dataset = load_dataset(tokenizer)
    return evaluate(model, tokenizer, dataset, "test")

def perform_experiment():
    for i in range(7):
        if not os.path.exists(f"saved_models/average_clf-{i}.pt"):
            train_classifier(f"-{i}")

    macro_scores = []
    micro_scores = []
    for i in range(7):
        _, macro_f1, micro_f1 = test_classifier(f"-{i}")
        macro_scores.append(macro_f1)
        micro_scores.append(micro_f1)

    print(f"Average Macro F1: {np.mean(macro_scores):.3f} ({[f'{x:.3f}' for x in macro_scores]})")
    print(f"Average Micro F1: {np.mean(micro_scores):.3f} ({[f'{x:.3f}' for x in micro_scores]})")
