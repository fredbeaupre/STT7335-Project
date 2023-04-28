import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def compute_accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    correct = np.sum(y_pred == y_true)
    return correct / y_pred.shape[0]


def compute_recall(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    tp = 0
    fn = 0
    for p, t in zip(y_pred, y_true):
        if p == 1 and t == 1:
            tp += 1
        elif p == 0 and t == 1:
            fn += 1
    return (tp) / (tp + fn + 1e-6)


def compute_precision(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    tp = 0
    fp = 0
    for p, t in zip(y_pred, y_true):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
    return (tp) / (tp + fp + 1e-6)


def compute_tpr(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    tp = 0
    fn = 0
    for p, t in zip(y_pred, y_true):
        if p == 1 and t == 1:
            tp += 1
        elif p == 0 and t == 1:
            fn += 1
    return (tp) / (tp + fn + 1e-6)


def compute_fpr(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    fp = 0
    tn = 0
    for p, t in zip(y_pred, y_true):
        if p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
    return (fp) / (fp + tn + 1e-6)


def compute_roc(raw_pred, y_true, save_dir="./saved_models/data2vec_balanced_classification"):
    fpr, tpr, thresholds = roc_curve(y_true, raw_pred)
    fig = plt.figure()
    plt.plot(fpr, tpr, color="tab:blue", marker='.')
    plt.xlabel('FPR')
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    fig.savefig('{}/roc_curve.png'.format(save_dir))
    fig.savefig('{}/roc_curve.pdf'.format(save_dir),
                bbox_inches='tight', transparent=True)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def m_grouping(scores, labels, m=5):
    sorted_labels = [x for _, x in sorted(zip(scores, labels), reverse=True)]
    total_pos = sum(sorted_labels)
    scores = sorted(scores, reverse=True)
    groups = list(zip(scores, sorted_labels))
    groups = list(split(groups, m))
    return groups, total_pos


def response_rate_one_group(group, threshold=0.5):
    pos_observations = len([item for item in group if item[1] == 1])
    pos_responses = 0
    for instance in group:
        score, label = instance[0], instance[1]
        if score > threshold and label == 1:
            pos_responses += 1
    print(
        f"Positive observations {pos_observations}\tPositive responses {pos_responses}")
    if pos_observations == 0:
        return 0
    else:
        return pos_responses / pos_observations


def compute_response_rate(scores, labels, threshold):
    groups, _ = m_grouping(scores, labels)
    response_rates = []
    for group in groups:
        rate = response_rate_one_group(group, threshold + 1)
        response_rates.append(rate)
    return response_rates


def compute_lift(scores, labels):
    big_group = list(zip(scores, labels))
    cumul_response = response_rate_one_group(big_group)
    groups, _ = m_grouping(scores, labels)
    print(groups)
    response_rates = []
    lifts = []
    for group in groups:
        rate = response_rate_one_group(group)
        response_rates.append(rate)
        lift = rate / cumul_response
        lifts.append(lift)
    return lifts, response_rates
