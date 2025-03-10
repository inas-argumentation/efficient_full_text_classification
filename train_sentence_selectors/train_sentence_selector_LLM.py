import json
import os.path
import sys
import itertools
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from train_sentence_selectors.sentence_selection_data_and_eval import LLMSentenceDataset, collate_fn, evaluate_sentence_classifier, load_LLM_dataset, load_importance_dataset, load_entropy_dataset
from settings import data_dir
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.load_data import load_dataset
from blingfire import text_to_sentences
from settings import model_type, device

def load_llm_dataset(raw_dataset, tokenizer):
    if os.path.exists(data_dir("LLM_sentence_assessment_new.json")):
        with open(data_dir("LLM_sentence_assessment_new.json"), "r") as f:
            created_dataset = json.load(f)
    else:
        created_dataset = {"train": [], "val": [], "test": []}

    if len(created_dataset["train"]) == len(raw_dataset["train"]) and \
            len(created_dataset["val"]) == len(raw_dataset["val"]) and \
            len(created_dataset["test"]) == len(raw_dataset["test"]):
        return created_dataset

    from transformers import pipeline
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype=torch.bfloat16)

    for split in ["train", "val", "test"]:
        for sample in tqdm(raw_dataset[split], file=sys.stdout):
            if len([x for x in created_dataset[split] if x["dataset_idx"] == sample["dataset_idx"]]) > 0:
                continue
            full_text = sample['text']
            sentences = text_to_sentences(full_text).split("\n")
            array = sample["evidence_array"]

            if array is None:
                labels = [-1] * len(tokenizer.tokenize(full_text))
            else:
                labels = []
                current_token_idx = 0
                for sentence in sentences:
                    pre_token_idx = current_token_idx
                    current_token_idx += len(tokenizer.tokenize(sentence))
                    label = 1 if np.sum(array[pre_token_idx:current_token_idx]) > 0 else 0
                    labels.append(label)

            sentence_assessments = []
            for i in range(len(sentences)):
                if len(sentences[i]) < 20:
                    continue

                sentences_before = " ".join(sentences[max(0, i-3):i]).strip()
                sentences_after = " ".join(sentences[i+1:i+4]).strip()
                sentences_before = "Not available." if len(sentences_before) < 5 else f"\"{sentences_before}\""
                sentences_after = "Not available." if len(sentences_after) < 5 else f"\"{sentences_after}\""

                prompt = f'''
                We want to create EICAT impact assessments, which contain overviews of scientific literature that reports impacts that an invasive species has.
                The original findings of each paper are used to determine the impact category. The following categories exist:
                
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
                
                Your task is to determine if a single sentence from a scientific paper provides information on the impact category that the species "{sample['species']}" has.
                Judge if the given sentence is "not useful", "slightly useful" or "very useful" to determine the correct impact category. You will receive the three sentences before and after the sentence you shall assess to have better knowledge of the context.
                
                These are the three sentences before the sentence you shall assess:
                {sentences_before}
                This is the sentence you shall assess:
                "{sentences[i]}"
                These are the sentences after the sentence you shall assess:
                {sentences_after}
                
                Use the following response format:
                Answer: [not useful/slightly useful/very useful]
                END.
                '''
#                Note that we are only interested in sentences that report on the original findings of the authors or direct interpretations of them, and not in summaries of previous work (which is usually indicated by a citation).

                messages = [{"role": "user", "content": prompt}]

                response = pipe(messages, num_return_sequences=1, top_k=1, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=20, max_length=None)
                output_string = response[0]["generated_text"][1]["content"].lower()
                try:
                    score = 2 if "very" in output_string else (1 if "slightly" in output_string else 0) #"#int(extract_score(output_string))
                    sentence_assessments.append((sentences[i], score))
                    #print(score, labels[i], sentences[i])
                    #print(output_string)
                except:
                    print(f"FAIL: {output_string}")
            created_dataset[split].append({"paper_id": sample["paper_id"],
                                           "dataset_idx": sample["dataset_idx"],
                                           "evidence_array": list(array) if array is not None else None,
                                           "LLM_assessments": sentence_assessments})
            with open(data_dir("LLM_sentence_assessment_new.json"), "w") as f:
                json.dump(created_dataset, f)
    return created_dataset


def train_sentence_classifier(batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    raw_dataset = load_dataset(tokenizer)
    LLM_dataset = load_llm_dataset(raw_dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=3).to(device)

    train_dataset = LLMSentenceDataset(LLM_dataset['train'], raw_dataset["train"], tokenizer)
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)))
    train_batch_generator = itertools.cycle(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"])["LLM"]

    epochs_without_improvement = 0
    loss_avg = None
    for epoch in range(200):
        model.train()

        for _ in (bar := tqdm(range(500), desc=f"Epoch {epoch}", file=sys.stdout)):
            inputs, labels = next(train_batch_generator)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels.to(torch.long))
            scale_mask = (labels == 0)
            loss[scale_mask] *= 0.2
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

        score = evaluate_sentence_classifier(model, tokenizer, raw_dataset["val"], LLM_dataset["val"])["LLM"]

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"saved_models/sentence_classifier_LLM.pt")
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
    model.load_state_dict(torch.load(f"saved_models/sentence_classifier_LLM.pt"))

    evaluate_sentence_classifier(model, tokenizer, raw_dataset["test"], LLM_dataset["test"], importance_dataset=importance_dataset["test"], entropy_dataset=entropy_dataset["test"])


if __name__ == "__main__":
    #train_sentence_classifier()
    test_sentence_selection_model()