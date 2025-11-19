import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_popularity_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the popularity prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    rmsle = np.around(np.sqrt(mean_squared_error(labels, predicts)), 4)

    msle = np.around(mean_squared_error(labels, predicts), 4)

    pred_mean, label_mean = np.mean(predicts, axis=0), np.mean(labels, axis=0)
    pre_std, label_std = np.std(predicts, axis=0), np.std(labels, axis=0)
    pcc = np.around(np.mean((predicts - pred_mean) * (labels - label_mean) / (pre_std * label_std), axis=0), 4)

    male = np.around(mean_absolute_error(labels, predicts), 4)

    label_p2 = np.power(2, labels)
    pred_p2 = np.power(2, predicts)
    result = np.mean(np.abs(np.log2(pred_p2 + 1) - np.log2(label_p2 + 1)) / np.log2(label_p2 + 2))
    mape = np.around(result, 4)

    return {'rmsle': rmsle, 'msle': msle, 'pcc': pcc, 'male': male, 'mape': mape}
