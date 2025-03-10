
def run_standard_encoder_classification_experiments():
    from text_classifiers import average_clf, modernBERT_clf
    average_clf.perform_experiment()
    modernBERT_clf.perform_experiment(False)
    modernBERT_clf.perform_experiment(True)

def train_and_text_sentence_selectors():
    from train_sentence_selectors import (train_sentence_selector_importance, train_sentence_selector_LLM,
                                          train_sentence_selector_entropy, train_sentence_selector_evidence)

    train_sentence_selector_importance.train_sentence_classifier()
    train_sentence_selector_LLM.train_sentence_classifier()
    train_sentence_selector_entropy.train_sentence_classifier()
    train_sentence_selector_evidence.train_sentence_classifier()

    train_sentence_selector_importance.test_sentence_selection_model()
    train_sentence_selector_LLM.test_sentence_selection_model()
    train_sentence_selector_entropy.test_sentence_selection_model()
    train_sentence_selector_evidence.test_sentence_selection_model()

def run_sentence_selection_encoder_experiments():
    from text_classifiers import selection_bert_clf
    selection_bert_clf.perform_experiment(sentence_selection="evidence", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="evidence", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="LLM", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="LLM", randomization=True)
    selection_bert_clf.perform_experiment(sentence_selection="importance", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="importance", randomization=True)
    selection_bert_clf.perform_experiment(sentence_selection="entropy", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="entropy", randomization=True)
    selection_bert_clf.perform_experiment(sentence_selection="random", randomization=False)
    selection_bert_clf.perform_experiment(sentence_selection="random", randomization=True)

def run_LLM_classification_experiment():
    from text_classifiers import LLM_clf
    LLM_clf.evaluate("test", sentence_extraction=None)  # Full text
    LLM_clf.evaluate("test", sentence_extraction="evidence", randomized=True)
    LLM_clf.evaluate("test", sentence_extraction="evidence", randomized=False)
    LLM_clf.evaluate("test", sentence_extraction="importance", randomized=True)
    LLM_clf.evaluate("test", sentence_extraction="importance", randomized=False)
    LLM_clf.evaluate("test", sentence_extraction="random", randomized=True)
    LLM_clf.evaluate("test", sentence_extraction="random", randomized=False)
    LLM_clf.evaluate("test", sentence_extraction="entropy", randomized=True)
    LLM_clf.evaluate("test", sentence_extraction="entropy", randomized=False)
    LLM_clf.evaluate("test", sentence_extraction="LLM", randomized=True)
    LLM_clf.evaluate("test", sentence_extraction="LLM", randomized=False)

if __name__ == '__main__':
    run_standard_encoder_classification_experiments()
    train_and_text_sentence_selectors()
    run_sentence_selection_encoder_experiments()
    run_LLM_classification_experiment()