# codin = utf-8
"""
Author: Ninghuyan
Email: huyanning@stu.xidian.edu.cn
Time 2018-07-18
Please email me if you find bugs, or have suggestions or questions!

Huyan N, Zhang X, Zhou H, et al. Hyperspectral Anomaly Detection via Background and
Potential Anomaly Dictionaries Construction[J].
IEEE Transactions on Geoscience and Remote Sensing, 2018.
"""


import numpy as np
# import HyperProTool as hyper
import scipy.io as sio
from LRSR import LRSR
#from LRSR_1 import LRSR
from dic_constr import dic_constr
from result_show import result_show
from ROC_AUC import ROC_AUC
import HyperProTool as hyper

# data pre-precessing
data = sio.loadmat("Sandiego.mat")
data3d = np.array(data["Sandiego"], dtype=float)
data3d = data3d[0:100, 0:100, :]
remove_bands = np.hstack((range(6), range(32, 35, 1), range(93, 97, 1), range(106, 113), range(152, 166), range(220, 224)))
data3d = np.delete(data3d, remove_bands, axis=2)
rows, cols, bands = data3d.shape
groundtruthfile = sio.loadmat("PlaneGT.mat")
groundtruth = np.array(groundtruthfile["PlaneGT"])
rows, cols, bands = data3d.shape

# background and anomaly dictionary construction
data2d, bg_dic, tg_dic,bg_dic_label, tg_dic_label = dic_constr(data3d, groundtruth, 3, 10, 10, 0.05, 200)

# low rank and sparse representaion
Z, E, S = LRSR(bg_dic, tg_dic, data2d, 0.001, 0.01)

# result visualization
background2d, target2d = result_show(bg_dic, tg_dic, Z, S, E, rows, cols, bands, bg_dic_label, tg_dic_label)

# ROC curve show
auc = ROC_AUC(target2d, groundtruth)
print("The AUC is: {0}".format(auc))


