import os.path
import time
import torch
from tqdm import tqdm
import random
import numpy as np
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.load_data import load_dataset
from auxiliary.split_texts import split_sample
from auxiliary.loss_fn import categorical_cross_entropy_with_logits
from auxiliary.evaluate_predictions import evaluate_classification_predictions
from blingfire import text_to_sentences
from collections import Counter
from settings import model_type, device

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def weighted_sample_without_replacement(indices, probabilities, n_samples):
    probabilities = probabilities / np.sum(probabilities)
    if n_samples > len(indices):
        raise ValueError("Cannot sample more items than available")

    exp_samples = np.random.exponential(1.0 / probabilities)
    selected_indices = np.argpartition(exp_samples, n_samples)[:n_samples]
    return indices[selected_indices]

def predict_sentence_scores(model, tokenizer, sample, batch_size=16):
    text = sample["text"]
    sentences = text_to_sentences(text).split("\n")

    if model is None:
        return list(zip(np.random.random(len(sentences)), sentences))

    inputs = []
    for idx in range(len(sentences)):
        sentence_tokens = len(tokenizer.tokenize(sentences[idx]))

        add_sentences_before = sentences[max(idx-3, 0):idx]
        while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
            add_sentences_before.pop(0)

        input_text = (sample["species"] + "; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentences[idx] +
              f" {tokenizer.sep_token} " + " ".join(sentences[idx+1:min(len(sentences), idx+4)]))
        inputs.append(input_text)

    sentence_predictions = []
    with torch.no_grad():
        for idx in range(0, len(inputs), batch_size):
            inputs = tokenizer(sentences[idx:idx + batch_size], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
            predictions = model(**inputs).logits
            if predictions.shape[-1] == 3:
                predictions = (torch.softmax(predictions, dim=-1) * torch.tensor([[0, 1, 2]], device=device, dtype=torch.float)).sum(-1)
            sentence_predictions += predictions.flatten().tolist()
    return list(zip(sentence_predictions, sentences))

def create_text(sentence_scores, random_scores, randomized):
    sentences = [x[1] for x in sentence_scores]
    scores = [x[0] for x in sentence_scores]

    if len(scores) <= 15:
        return " ".join(sentences)

    if not randomized:
        top_sentences = np.sort(np.argsort(scores)[-15:])
    else:
        if random_scores:
            random.shuffle(scores)
        potential_indices = np.argsort(scores)[-30:]
        probabilities = np.arange(1, len(potential_indices)+1)
        sample = weighted_sample_without_replacement(potential_indices, probabilities, 15)
        top_sentences = np.sort(sample)
    return " ".join([sentences[x] for x in top_sentences])

def create_one_hot_label(labels, num_classes=6):
    one_hot = torch.zeros(num_classes)
    for label in labels:
        one_hot[label] = 1.0
    return one_hot

def evaluate(model, sentence_model, tokenizer, dataset, split, random_scores, randomization):
    model.eval()
    all_predictions = []
    all_labels = []
    all_losses = []

    t = time.time()
    with torch.no_grad():
        for sample in tqdm(dataset[split], desc=f"Evaluating {split}"):
            if sentence_model is None and "sentence_scores" in sample:
                sentence_scores = sample["sentence_scores"]
            else:
                sentence_scores = predict_sentence_scores(sentence_model, tokenizer, sample)

            label = create_one_hot_label(sample['labels'])

            sampled_predictions = []
            sampled_losses = []
            for pred_idx in range(10 if randomization else 1):
                input_text = create_text(sentence_scores, random_scores, randomization)
                word_lists, _, _ = split_sample(tokenizer, input_text, 500, None, 50)
                inputs = tokenizer(word_lists, max_length=500, truncation=True, padding=True, return_tensors="pt").to(device)

                outputs = model(**inputs)
                avg_logits = torch.mean(outputs.logits, dim=0)
                prediction = torch.argmax(avg_logits, dim=-1).item()
                sampled_predictions.append(prediction)
                sampled_losses.append(categorical_cross_entropy_with_logits(avg_logits, label.to("cuda")).item())

            prediction = most_common(sampled_predictions)
            all_predictions.append(prediction)
            all_labels.append(label.numpy())
            all_losses.append(np.mean(sampled_losses))

    print("Eval time: ", time.time() - t)
    macro_f1, micro_f1 = evaluate_classification_predictions(np.array(all_predictions), np.array(all_labels), convert_predictions=True)
    avg_loss = np.mean(all_losses)
    print(f"{split} loss: {avg_loss:.3f}")
    return avg_loss, macro_f1, micro_f1

def train_classifier(dataset, randomization, save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    label_counts = torch.stack([create_one_hot_label(x["labels"]) for x in dataset["train"]], dim=0).sum(0)
    class_weights = torch.tensor((1 / torch.sqrt(label_counts)) / (1 / torch.sqrt(label_counts)).mean(), device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    epochs_without_improvement = 0
    batch_size = 32
    loss_avg = None

    score_func = lambda x: float(np.nan_to_num(x[1] + x[2] - x[0], nan=-np.inf))
    best_score = score_func(evaluate(model, None, tokenizer, dataset, 'val', "random" in save_name_end, randomization))
    print(f"Initial score: {best_score:.4f}")

    for epoch in range(200):
        model.train()
        total_loss = 0
        random.shuffle(dataset['train'])

        batch_texts = []
        batch_labels = []
        batch_splits = []
        for sample_idx, sample in enumerate(bar:= tqdm(dataset['train'], desc=f"Epoch {epoch}")):
            new_text = create_text(sample["sentence_scores"], "random" in save_name_end, randomized=randomization)
            word_lists, _, _ = split_sample(tokenizer, new_text, 500, None, 50)

            batch_texts += word_lists
            label = create_one_hot_label(sample['labels']).to(device)
            batch_labels.append(label)
            batch_splits.append(len(word_lists))

            if len(batch_texts) >= batch_size or sample_idx == len(dataset["train"])-1:

                inputs = tokenizer(batch_texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
                outputs = model(**inputs).logits

                repeated_labels = []
                for current_idx, n_splits in enumerate(batch_splits):
                    for _ in range(n_splits):
                        repeated_labels.append(batch_labels[current_idx])
                loss = categorical_cross_entropy_with_logits(outputs, torch.stack(repeated_labels, dim=0), class_weights.unsqueeze(0)).mean()

                if loss_avg is None:
                    loss_avg = loss.item()
                else:
                    loss_avg = 0.95 * loss_avg + 0.05 * loss.item()
                bar.desc = f"Epoch {epoch}, Loss: {loss_avg:.2f}"

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

        score = score_func(evaluate(model, None, tokenizer, dataset, 'val', "random" in save_name_end, randomization))
        print(f"Validation score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/sampled_clf{save_name_end}.pt")
            epochs_without_improvement = 0
            print("New best! Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print("Early stopping triggered")
            break

def test_classifier(sentence_model, randomization, save_name_end=""):
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=6).to(device)
    model.load_state_dict(torch.load(f"saved_models/sampled_clf{save_name_end}.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    dataset = load_dataset(tokenizer)
    return evaluate(model, sentence_model, tokenizer, dataset, "test", "random" in save_name_end, randomization)

# sentence_selection shall be one of "evidence", "LLM", "entropy", "importance", "random"
def perform_experiment(sentence_selection=None, randomization=False):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    dataset = load_dataset(tokenizer)

    if sentence_selection not in ["random"]:
        sentence_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            num_labels=1 if sentence_selection == "evidence" else 3).to(device)
        sentence_model.load_state_dict(torch.load(f"saved_models/sentence_classifier_{sentence_selection}.pt"))
        sentence_model.eval()
    else:
        sentence_model = None

    for i in range(7):
        if not os.path.exists(f"saved_models/sampled_clf-{i}-{sentence_selection}-{randomization}.pt"):
            train_classifier(dataset, randomization, f"-{i}-{sentence_selection}-{randomization}")

    macro_scores = []
    micro_scores = []
    for i in range(7):
        _, macro_f1, micro_f1 = test_classifier(sentence_model, randomization, f"-{i}-{sentence_selection}-{randomization}")
        macro_scores.append(macro_f1)
        micro_scores.append(micro_f1)
    print(f"Selection: {sentence_selection}, Randomization: {randomization}")
    print(f"Average Macro F1: {np.mean(macro_scores):.3f} ({[f'{x:.3f}' for x in macro_scores]})")
    print(f"Average Micro F1: {np.mean(micro_scores):.3f} ({[f'{x:.3f}' for x in micro_scores]})")


if __name__ == "__main__":
    perform_experiment(sentence_selection="evidence", randomization=False)
    #perform_experiment(sentence_selection="evidence", randomization=False)
    #perform_experiment(sentence_selection="LLM", randomization=False)
    #perform_experiment(sentence_selection="LLM", randomization=True)
    #perform_experiment(sentence_selection="importance", randomization=False)
    #perform_experiment(sentence_selection="importance", randomization=True)
    #perform_experiment(sentence_selection="entropy", randomization=False)
    #perform_experiment(sentence_selection="entropy", randomization=True)
    #perform_experiment(sentence_selection="random", randomization=False)
    #perform_experiment(sentence_selection="random", randomization=True)