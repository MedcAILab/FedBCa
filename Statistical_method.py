from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_AUC(pred_value, label_value):
    "MinMaxScaler"
    scaler = MinMaxScaler(feature_range=(0, 1))
    pred_value = scaler.fit_transform(pred_value)

    if (sum(label_value) == len(pred_value)) or (sum(label_value) == 0):
        AUC = 0
        print('only one class')
    else:
        AUC = metrics.roc_auc_score(label_value, pred_value)

    return AUC


def get_best_threshold(pred_value, label_value):
    # 计算最佳阈值
    fpr, tpr, thresholds = metrics.roc_curve(label_value, pred_value)
    # 计算约登指数
    Youden_index = tpr + (1 - fpr)
    best_threshold = thresholds[Youden_index == np.max(Youden_index)][0]

    # have no idea about that threshold is bigger than 1 sometimes
    # maybe can find in https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    # or https://github.com/scikit-learn/scikit-learn/issues/3097
    if best_threshold > 1:
        best_threshold = 0.5

    return best_threshold


def get_accuracy(pred_value, label_value):
    scaler = MinMaxScaler(feature_range=(0, 1))
    pred_value = scaler.fit_transform(np.array(pred_value).reshape(-1, 1))
    threshold = get_best_threshold(pred_value, label_value)

    pred_value_list = [1 if pred_value_i >= threshold else 0 for pred_value_i in pred_value]

    TP = sum([1 if pred_value_i == 1 and label_value_i == 1 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    FN = sum([1 if pred_value_i == 0 and label_value_i == 1 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    TN = sum([1 if pred_value_i == 0 and label_value_i == 0 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    FP = sum([1 if pred_value_i == 1 and label_value_i == 0 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    ACC = float(TP + TN) / (float(TP + TN + FP + FN) + 1e-6)

    return ACC


def get_sensitivity(pred_value, label_value):
    scaler = MinMaxScaler(feature_range=(0, 1))
    pred_value = scaler.fit_transform(np.array(pred_value).reshape(-1, 1))
    threshold = get_best_threshold(pred_value, label_value)
    pred_value_list = [1 if pred_value_i >= threshold else 0 for pred_value_i in pred_value]

    "TP : True Positive"
    "FN : False Negative"

    TP = sum([1 if pred_value_i == 1 and label_value_i == 1 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    FN = sum([1 if pred_value_i == 0 and label_value_i == 1 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    SE = float(TP) / (float(TP + FN) + 1e-6)

    return SE


def get_specificity(pred_value, label_value):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # pred_value = scaler.fit_transform(pred_value)
    pred_value = scaler.fit_transform(np.array(pred_value).reshape(-1, 1))
    threshold = get_best_threshold(pred_value, label_value)
    pred_value_list = [1 if pred_value_i > threshold else 0 for pred_value_i in pred_value]

    "# TN : True Negative"
    "# FP : False Positive"

    TN = sum([1 if pred_value_i == 0 and label_value_i == 0 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    FP = sum([1 if pred_value_i == 1 and label_value_i == 0 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    SP = float(TN) / (float(TN + FP) + 1e-6)

    return SP


def get_precision(pred_value, label_value):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # pred_value = scaler.fit_transform(pred_value)
    pred_value = scaler.fit_transform(np.array(pred_value).reshape(-1, 1))
    threshold = get_best_threshold(pred_value, label_value)
    pred_value_list = [1 if pred_value_i > threshold else 0 for pred_value_i in pred_value]

    "# TP : True Positive"
    "# FP : False Positive"

    TP = sum([1 if pred_value_i == 1 and label_value_i == 1 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    FP = sum([1 if pred_value_i == 1 and label_value_i == 0 else 0
              for pred_value_i, label_value_i in zip(pred_value_list, label_value)])

    PC = float(TP) / (float(TP + FP) + 1e-6)

    return PC
