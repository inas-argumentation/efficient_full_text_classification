### This is the official code repository for the paper [Efficient Scientific Full Text Classification: The Case of EICAT Impact Assessments](https://arxiv.org/abs/2502.06551) by Marc Felix Brinner and Sina Zarrieß published at NLP4Ecology 2025.

# Setup

We ran our experiments using Python 3.10. Please install the packages specified in the `requirements.txt` file.

# Data

Our experiments center around classifying scientific full text papers addressing the impact that a range of invasive species have on ecosystems.

The file `data/EICAT_fulltext_dataset_final.json` lists all species that were addressed in our study, as well as the corresponding paper references, paper-specific labels as well as the evidence sentences that were extracted from the EICAT impact assessment files.
Due to copyright reasons, we are not able to share the dataset of full-text papers publicly. Please reach out to `marc.brinner@uni-bielefeld.de` to check if we can provide the data for research purposes. If you aquired the full-texts, please place them in a folder `data/EICAT_papers_json`, with each paper having a separate JSON file (named INDEX.json, according to the index that is assigned to each paper in our dataset). Each json file shall contain the fields "title", "abstract" and "body" that provided these three components of the paper.

# Repository Structure

The repository is structured as follows:

* `main.py` can be used to run all experiments. Please uncomment lines corresponding to experiments that you are not interested in.
* `text_classifiers` contains the code for training and evaluating all classification models (both encoder- and LLM-based). For all experiments that require sentence-selector models, these must be trained first.
* `train_sentence_selectors` contains the code for training sentence selection models. For entropy and importance selectors, three standard encoder classifiers need to be trained first, since they are used as basis for sentence assessment.
* `data` contains the data used in the experiments as well as code for loading it. The data included in this repo is the following:
    * `EICAT_fulltext_dataset_final.json` contains the dataset as described above.
    * `BERT_entropy_dataset.json` and `BERT_importance_sentence_assessment.json` are the sentence assessments created in our experiments for the "entropy" and "importance" strategies.
    * `LLM_sentence_assessment_new.json` are the sentence assessments created by Llama-3.1 8B.
    * `full_text_evidence.json` contains the sentences from the full-texts that we matched to the evidence extracted by the human annotators.

# Citation

At the time of writing, the proceedings of the workshop are not published. For the time being, please cite:

```bibtex
@misc{brinner2025efficientscientifictextclassification,
      title={Efficient Scientific Full Text Classification: The Case of EICAT Impact Assessments}, 
      author={Marc Felix Brinner and Sina Zarrieß},
      year={2025},
      eprint={2502.06551},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06551}, 
}
