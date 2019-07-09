# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import HyperProTool as hyper


def ROC_AUC(target2d, groundtruth):
    """
    
    :param target2d: the 2D anomaly component  
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    # result = hyper.hypernorm(result, "minmax")
    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    plt.figure(2)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return auc
