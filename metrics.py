import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def compute_accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    correct = np.sum(y_pred == y_true)
    return correct / y_pred.shape[0]


def compute_recall(y_pred, y_true):
    tp = np.sum(y_pred == 1 and y_true == 1)
    fn = np.sum(y_pred == 0 and y_true == 1)
    return (tp) / (tp + fn + 1e-6)


def compute_precision(y_pred, y_true):
    tp = np.sum(y_pred == 1 and y_true == 1)
    fp = np.sum(y_pred == 1 and y_true == 0)
    return (tp) / (tp + fp + 1e-6)


def compute_tpr(y_pred, y_true):
    tp = np.sum(y_pred == 1 and y_true == 1)
    fn = np.sum(y_pred == 0 and y_true == 1)
    return (tp) / (tp + fn + 1e-6)


def compute_fpr(y_pred, y_true):
    fp = np.sum(y_pred == 1 and y_true == 0)
    tn = np.sum(y_pred == 0 and y_true == 0)


def compute_roc(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)


def compute_response():
    pass


def compute_lift():
    pass
