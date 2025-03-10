import json
import os.path
import numpy as np
from tqdm import tqdm
from settings import data_dir
from blingfire import text_to_sentences

def load_evidence_sentences(processed_data):
    with open(data_dir("full_text_evidence.json"), "r") as f:
        evidence_sentences = json.load(f)
        return {int(x): y for x, y in evidence_sentences.items()}

# Create arrays that encode which sentences are evidence sentences.
def create_ground_truth_arrays(dataset, evidence_sentences, tokenizer):
    for split in dataset:
        for idx in range(len(dataset[split])):
            paper_full_text = dataset[split][idx]["text"]
            dataset_idx = dataset[split][idx]["dataset_idx"]

            if (dataset_idx not in evidence_sentences or len(evidence_sentences[dataset_idx]) == 0 or sum([len(x) for x in evidence_sentences[dataset_idx]]) <= 5):
                dataset[split][idx]["evidence_array"] = None
                continue

            all_sentences = text_to_sentences(paper_full_text).split("\n")
            sentence_spans = []
            current_index = 0
            for sentence in all_sentences:
                found_idx = paper_full_text.find(sentence, current_index)
                if found_idx < 0:
                    raise Exception("Sentence not found!")
                sentence_spans.append((found_idx, found_idx + len(sentence), sentence))
                current_index = found_idx + len(sentence)

            paper_evidence = set(evidence_sentences[dataset_idx])
            paper_evidence = sorted([(paper_full_text.find(e), e) for e in paper_evidence], key=lambda x: x[0])

            if paper_evidence[0][0] < 0:
                raise Exception("Evidence sentence not found in text")

            evidence_spans = [(x, x+len(y)) for x, y in paper_evidence]

            all_sentences_annotated = []
            for start, end, sentence in sentence_spans:
                for e_start, e_end in evidence_spans:
                    if (e_start < end and e_end > start):
                        all_sentences_annotated.append([start, end, sentence, 1])
                        break
                if len(all_sentences_annotated) == 0 or all_sentences_annotated[-1][0] != start:
                    all_sentences_annotated.append([start, end, sentence, 0])

            tokenizer_output = tokenizer(paper_full_text, add_special_tokens=False)
            array = [0] * len(tokenizer_output.tokens())

            # For each token, check if it overlaps with any evidence span
            for token_idx in range(len(tokenizer_output.tokens())):
                token_span = tokenizer_output.token_to_chars(token_idx)
                if token_span is None:
                    continue

                for ev_start, ev_end in evidence_spans:
                    if (token_span.start < ev_end and token_span.end > ev_start):
                        array[token_idx] = 1
                        break

            dataset[split][idx]["evidence_array"] = np.array(array, dtype="float")
            dataset[split][idx]["evidence_sentences"] = list(paper_evidence)
            dataset[split][idx]["evidence_sentence_annotation"] = all_sentences_annotated

            if sum(array) == 0:
                print(f"Warning: No evidence tokens found for document {dataset_idx}")

    return dataset


def load_paper_full_text(paper_id):
    with open(data_dir(f"EICAT_papers_json/{paper_id}.json")) as f:
        content = json.load(f)
    title = content["title"].strip() if content["title"] is not None else ""
    abstract = content["abstract"].strip() + " " if content["abstract"] is not None else ""
    text = content["body"].strip() if content["body"] is not None else ""

    if len(title) > 0:
        if title[-1] not in ["!", "?", "."]:
            title += ". "
        else:
            title += " "

    return f"{title} {abstract} {text}"


def load_dataset(tokenizer, return_label_dict=False):
    with open(data_dir("EICAT_fulltext_dataset_final.json"), "r") as f:
        data = json.load(f)

    label_dict = {}
    def get_label(label):
        label = label[:2]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        return label_dict[label]

    processed_data = {'train': [], 'val': [], 'test': []}

    dataset_idx = 0
    for split in ['train', 'val', 'test']:
        for species, papers in data[split].items():
            for entry in papers.values():
                processed_data[split].append({'species': species,
                                              'paper_id': entry["paper_id"],
                                              'labels': [get_label(x) for x in entry["label"]],
                                              'text': load_paper_full_text(entry["paper_id"]),
                                              'evidence': entry['evidence'],
                                              'dataset_idx': dataset_idx})
                dataset_idx += 1

    evidence_sentences = load_evidence_sentences(processed_data)
    processed_data = create_ground_truth_arrays(processed_data, evidence_sentences, tokenizer)

    if return_label_dict:
        return processed_data, label_dict
    return processed_data
