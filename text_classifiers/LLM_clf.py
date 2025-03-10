import os
import random
import torch
import numpy as np
import time
from tqdm import tqdm
from collections import Counter
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from data.load_data import load_dataset
from auxiliary.evaluate_predictions import evaluate_classification_predictions
from blingfire import text_to_sentences
from settings import device, LLM_checkpoint

checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

pipe = pipeline("text-generation",
                    model=LLM_checkpoint,
                    device_map="auto",
                    torch_dtype=torch.bfloat16)

def create_one_hot_label(labels, num_classes=6):
    one_hot = torch.zeros(num_classes)
    for label in labels:
        one_hot[label] = 1.0
    return one_hot

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

def get_llm_prediction(text, species, pipe, label_dict, prompt_start="full"):
    if prompt_start == "full":
        prompt_start = "This is a scientific paper about an invasive species: "
    elif prompt_start == "subset":
        prompt_start = "These are sentences that were extracted from a scientific paper about an invasive species (missing sentences are indicated by [...]): "

    prompt = prompt_start + f'''
    {text}

    This is the end of the scientific text.
    Your task is to classify the impact that the invasive species "{species}" has. Note that the text might contain information on other species. Possible classes are the following:
    1. Minimal:
    A taxon is considered to have impacts of Minimal Concern when it causes negligible levels of
    impacts, but no reduction in performance of individuals in the native biota. Note that all alien taxa
    have impacts on the recipient environment at some level, for example by altering species diversity or
    community similarity (e.g., biotic homogenisation), and for this reason there is no category equating
    to “no impact”. Only taxa for which changes in the individual performance of natives have been
    studied but not detected are assigned an MC category. Taxa that have been evaluated under the
    EICAT process but for which impacts have not been assessed in any study should not be classified in
    this category, but rather should be classified as Data Deficient.
    2. Minor
    A taxon is considered to have Minor impacts when it causes reductions in the performance of
    individuals in the native biota, but no declines in native population sizes, and has no impacts that
    would cause it to be classified in a higher impact category.
    3. Moderate
    A taxon is considered to have Moderate impacts when it causes declines in the population size of
    at least one native taxon, but has not been observed to lead to the local extinction of a native taxon.
    4. Major
    A taxon is considered to have Major impacts when it causes community changes through the local
    or sub-population extinction (or presumed extinction) of at least one native taxon, that would be
    naturally reversible if the alien taxon was no longer present. Its impacts do not lead to naturally
    irreversible local population, sub-population or global taxon extinctions.
    5. Massive
    A taxon is considered to have Massive impacts when it causes naturally irreversible community
    changes through local, sub-population or global extinction (or presumed extinction) of at least one
    native taxon.
    6. Data Deficient
    A taxon is categorised as Data Deficient when the best available evidence indicates that it has (or
    had) individuals existing in a wild state in a region beyond the boundary of its native geographic
    range, but either there is inadequate information to classify the taxon with respect to its impact, or
    insufficient time has elapsed since introduction for impacts to have become apparent. It is expected
    that all introduced taxa will have an impact at some level, because by definition an alien taxon in a
    new environment has a nonzero impact. However, listing a taxon as Data Deficient recognises that
    current information is insufficient to assess that level of impact.

    Return just the classification and end your answer, and provide one of the following labels as answer: "Minimal", "Minor", "Moderate", "Major", "Massive", "Data Deficient".
    Provide your answer by just using the following response format, and do not answer anything else in addition to that:
    Summary: [One sentence summarizing the key information that you consider for the assessment]
    Answer: [Your answer, that is one of the six labels]
    END.
    '''

    messages = [{"role": "user", "content": prompt}]
    response = pipe(messages, num_return_sequences=1, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=200,
                    do_sample=False, top_p=None, temperature=None)
    output_string = response[0]["generated_text"][1]["content"].lower()

    if "answer: minimal" in output_string:
        return label_dict["MC"]
    if "answer: minor" in output_string:
        return label_dict["MN"]
    if "answer: moderate" in output_string:
        return label_dict["MO"]
    if "answer: major" in output_string:
        return label_dict["MR"]
    if "answer: massive" in output_string:
        return label_dict["MV"]
    if "answer: data deficient" in output_string:
        return label_dict["DD"]
    print(f"Failed to parse LLM output: {output_string}")
    return -1

def get_sentence_scores(sentence_model, tokenizer, sentences, species):
    inputs = []
    for idx in range(len(sentences)):
        sentence_tokens = len(tokenizer.tokenize(sentences[idx]))

        add_sentences_before = sentences[max(idx-3, 0):idx]
        while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
            add_sentences_before.pop(0)

        input_text = (species + "; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentences[idx] +
              f" {tokenizer.sep_token} " + " ".join(sentences[idx+1:min(len(sentences), idx+4)]))
        inputs.append(input_text)

    sentence_predictions = []
    with torch.no_grad():
        for idx in range(0, len(inputs), 16):
            inputs = tokenizer(sentences[idx:idx + 16], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
            predictions = sentence_model(**inputs).logits
            if predictions.shape[-1] == 3:
                predictions = (torch.softmax(predictions, dim=-1) * torch.tensor([[0, 1, 2]], device=device, dtype=torch.float)).sum(-1)
            sentence_predictions += predictions.flatten().tolist()
    return sentence_predictions

def create_text(sentences, scores, randomized):
    if len(scores) <= 15:
        return " ".join(sentences)

    if not randomized:
        top_sentences = np.sort(np.argsort(scores)[-15:])
    else:
        potential_indices = np.argsort(scores)[-30:]
        probabilities = np.arange(1, len(potential_indices)+1)
        sample = weighted_sample_without_replacement(potential_indices, probabilities, 15)
        top_sentences = np.sort(sample)
    new_text = sentences[top_sentences[0]]
    prev_idx = top_sentences[0]
    for idx in top_sentences[1:]:
        if abs(idx - prev_idx) > 1:
            new_text += " [...]"
        new_text += " " + sentences[idx]
    return new_text

def evaluate(split, sentence_extraction=None, randomized=False):
    bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    dataset, label_dict = load_dataset(bert_tokenizer, return_label_dict=True)

    print(label_dict)

    if sentence_extraction is not None and sentence_extraction != "random":
        sentence_model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
            num_labels=1 if sentence_extraction == "evidence" else 3).to(device)
        sentence_model.load_state_dict(torch.load(f"saved_models/sentence_classifier_{sentence_extraction}.pt"))
        sentence_model.eval()

    all_predictions = []
    all_labels = []
    correct_counter = 0
    overall_counter = 0
    start_time = time.time()

    with torch.no_grad():
        for sample in (bar := tqdm(dataset[split], desc=f"Evaluating {split}")):
            full_text = sample['text']
            label = create_one_hot_label(sample['labels'])

            if sentence_extraction is not None:
                sentences = text_to_sentences(full_text).split("\n")
                if sentence_extraction == "random": scores = np.random.random(len(sentences))
                else: scores = get_sentence_scores(sentence_model, bert_tokenizer, sentences, sample["species"])

            sampled_predictions = []
            for pred_idx in range(10 if randomized and sentence_extraction is not None else 1):
                if sentence_extraction is not None:
                    full_text = create_text(sentences, scores, randomized)

                prediction = get_llm_prediction(full_text, sample["species"], pipe, label_dict, prompt_start="full" if sentence_extraction is None else "subset")
                sampled_predictions.append(prediction)
            #print(sampled_predictions)
            prediction = most_common(sampled_predictions)

            pred_one_hot = torch.zeros(6)
            if prediction >= 0:
                pred_one_hot[prediction] = 1.0
                correct_counter += label[prediction]
            overall_counter += 1
            bar.desc = f"Evaluating {split} (Prec: {correct_counter / overall_counter:.3f})"

            all_predictions.append(prediction)
            all_labels.append(label.numpy())
            #label = np.argmax(label.numpy())
            #label_code = [x for x in label_dict if label_dict[x] == label][0]
            # print(f"Real label: {label_code}")
            # print()
            # print()

    total_time = time.time() - start_time

    macro_f1, micro_f1 = evaluate_classification_predictions(np.array(all_predictions), np.array(all_labels), convert_predictions=True)

    print(f"{split} Results:")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Total running time: {total_time}")
    return macro_f1, micro_f1

# {'DD': 0, 'MR': 1, 'MO': 2, 'MN': 3, 'MC': 4, 'MV': 5}
if __name__ == "__main__":
    # evaluate("test", sentence_extraction="evidence", randomized=False)
    # evaluate("test", sentence_extraction="importance", randomized=True)
    evaluate("test", sentence_extraction="importance", randomized=False)
    # evaluate("test", sentence_extraction="evidence")
    #evaluate("test", sentence_extraction="entropy")
    # evaluate("test", sentence_extraction=None)