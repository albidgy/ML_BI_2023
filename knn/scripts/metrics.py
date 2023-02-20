import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    if y_pred.shape[0] > 0 and y_true.shape[0] > 0:
        tp = sum((y_pred == y_true) & (y_pred == 1))
        fp = sum((y_pred != y_true) & (y_pred == 1))
        tn = sum((y_pred == y_true) & (y_pred == 0))
        fn = sum((y_pred != y_true) & (y_pred == 0))

        if tp + fp == 0:
            return 0, 0, 0, 0

        if tp + fn == 0:
            return 0, 0, 0, 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        return precision, recall, f1, accuracy

    else:
        raise Exception('recheck your data')



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    if y_pred.shape[0] == y_true.shape[0] and y_true.shape[0] > 0:
        true_detect = sum(y_pred == y_true)
        accuracy = true_detect / (y_true.shape[0])
        return accuracy

    else:
        raise Exception('recheck your data')




def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return r2



def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    n = y_true.size
    mse = np.sum((y_true - y_pred)**2) / n
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    n = y_true.size
    mae = np.sum(np.abs(y_true - y_pred)) / n
    return mae
    