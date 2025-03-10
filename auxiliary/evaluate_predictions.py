import numpy as np

from auxiliary.split_texts import split_sample_and_return_words

span_score_names = {0: "Average token auc",
                    1: "Average token F1",
                    2: "Discrete token F1",
                    3: "Average span IoU F1",
                    4: "Discrete span IoU F1"}

def evaluate_classification_predictions(y_pred, y_true, print_statistics=True, convert_predictions=True):
    num_classes = y_true.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        if convert_predictions:
            one_hot_pred = np.zeros((len(y_pred), num_classes))
            one_hot_pred[np.arange(len(y_pred)), y_pred] = 1

            one_hot_gt = y_true
        else:
            one_hot_pred = y_pred
            one_hot_gt = y_true

        gt_counts = np.sum(one_hot_gt, axis=0)
        predicted_counts = np.sum(one_hot_pred, axis=0)
        actually_there = (gt_counts > 0).astype("int32")

        tp = np.sum(one_hot_pred * one_hot_gt, axis=0)
        fp = np.sum(one_hot_pred * (1-one_hot_gt), axis=0)
        tn = np.sum((1-one_hot_pred) * (1-one_hot_gt), axis=0)
        fn = np.sum((1-one_hot_pred) * one_hot_gt, axis=0)

        micro_precision = tp.sum() / (tp.sum() + fp.sum())
        micro_recall = tp.sum() / (fn.sum() + tp.sum())
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        precision = tp / (tp + fp)
        recall = tp / (fn + tp)
        f1 = 2 * precision * recall / (precision + recall)
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0
        f1[np.isnan(f1)] = 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if print_statistics:
            print("\nPer class metrics:")
            print("  Class | Precision  |  Recall  |    F1   |  Accuracy  |  Count")
            for c in range(num_classes):
                if actually_there[c] > 0:
                    print(f"   {c:2d}   |   {precision[c]:.3f}    |  {recall[c]:.3f}   |  {f1[c]:.3f}  |   {accuracy[c]:.3f}    |  {gt_counts[c]}/{predicted_counts[c]}")
                else:
                    print(f"    -   |     -      |    -     |    -    |     -      |  {gt_counts[c]}/{predicted_counts[c]}")


        macro_f1 = np.sum(f1*actually_there) / np.sum(actually_there)
        if print_statistics:
            print(f"Macro F1: {macro_f1:.3f}  Micro F1: {micro_f1:.3f}")

    return macro_f1, micro_f1