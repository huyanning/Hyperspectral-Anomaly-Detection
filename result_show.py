# coding=utf-8
import numpy as np
import HyperProTool as hyper
import matplotlib.pyplot as plt
import scipy.io as sio


def result_show(bg_dic, tg_dic, Z, S, E, rows, cols, bands, bg_dic_label, tg_dic_label):
    """
    
    :param bg_dic: the background dictionary
    :param tg_dic: the anomaly dictionary
    :param Z: the low rank coefficients
    :param S: the sparse coefficients
    :param E: the noise
    :param rows: the number of rows in original hyperpsectral image
    :param cols: the number of cols in original hypersepctral image 
    :param bands:  the number of bands in original hyperpsectral image
    :param bg_dic_label:  the index of the atoms in background dictionary
    :param tg_dic_label: the index of the atoms in anomaly dictionary
    :return: background2d: the 2D background component
                target2d : the 2D anomaly component
    
    """
    background2d = np.dot(bg_dic, Z)
    background3d = hyper.hyperconvert3d(background2d, rows, cols, bands)
    target2d = np.dot(tg_dic, S)
    target3d = hyper.hyperconvert3d(target2d, rows, cols, bands)
    noise3d = hyper.hyperconvert3d(E, rows, cols, bands)
    bg_dic_show = np.zeros((1, rows * cols))
    tg_dic_show = np.zeros((1, rows * cols))
    bg_dic_show[0, bg_dic_label] = 1
    tg_dic_show[0, tg_dic_label] = 1
    bg_dic_show = bg_dic_show.reshape(rows, cols)
    tg_dic_show = tg_dic_show.reshape(rows, cols)
    cluster_assment_file = sio.loadmat("cluster_assment.mat")
    cluster_assment = np.array(cluster_assment_file["cluster_assment"])
    label = cluster_assment
    label = label.transpose()
    segm_show = label.reshape(rows, cols)

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(background3d.mean(2))
    plt.xlabel('Background')
    plt.subplot(2, 3, 2)
    plt.imshow(target3d.mean(2))
    plt.xlabel('Anomaly')
    plt.subplot(2, 3, 3)
    plt.imshow(noise3d.mean(2))
    plt.xlabel('Noise')
    plt.subplot(2, 3, 4)
    plt.imshow(bg_dic_show)
    plt.xlabel('Background dictionary')
    plt.subplot(2, 3, 5)
    plt.imshow(tg_dic_show)
    plt.xlabel('Anomaly dictionary')
    plt.subplot(2, 3, 6)
    plt.imshow(segm_show)
    plt.xlabel('Segmentation')

    plt.show()
    return background2d, target2d
