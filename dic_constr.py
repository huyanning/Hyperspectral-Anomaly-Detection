# coding=utf-8
import numpy as np
import HyperProTool as hyper
import scipy.io as sio
from sklearn import decomposition


def dic_constr(data3d, groundtruth, win_size, cluster_num, K, selected_dic_percent, target_dic_num):
    """
    :param data3d: the original 3D hyperpsectral image 
    :param groundtruth:  a 2D matrix reflect the label of corresponding pixels 
    :param win_size: the size of window, such as 3X3, 5X5, 7X7
    :param cluster_num: the number of classters such as 5, 10, 15, 20
    :param K: the level of sparsity
    :param selected_dic_percent: the selected percent of the atoms to build the background dictionary
    :param target_dic_num: the selected number to build the anomaly dictionary
    :return: data2d:  the normalized data
             bg_dic:  the background dictionary
             tg_dic:  the anomaly dictionary
             bg_dic_ac_label:  the index of background dictionary atoms 
             tg_dic_label: the index of anomaly dictionary atoms      
    """
    data2d = hyper.hyperconvert2d(data3d)
    rows, cols, bands = data3d.shape
    data2d = hyper.hypernorm(data2d, "L2_norm")
    sio.savemat("data2d.mat", {'data2d': data2d})
    data3d = hyper.hyperconvert3d(data2d, rows, cols, bands)
    pca = decomposition.PCA(n_components=20, copy=True, whiten=False)
    dim_data = pca.fit_transform(data2d.transpose())
    data3d_dim = hyper.hyperconvert3d(dim_data.transpose(), rows, cols, 10)
    win_dim = hyper.hyperwincreat(data3d_dim, win_size)
    cluster_assment = hyper.Kmeans_win(win_dim, cluster_num)
    sio.savemat("cluster_assment.mat", {'cluster_assment': cluster_assment})
    win_matrix = hyper.hyperwincreat(data3d, win_size)
    sio.savemat("win_matrix.mat", {'win_matrix': win_matrix})
    wm_rows, wm_cols, wm_n = win_matrix.shape
    resdiual_stack = np.zeros((bands, win_size * win_size, wm_n))
    save_num = 0
    bg_dic_tuple = []
    bg_dic_ac_tuple = []
    bg_dic_fc_tuple = []
    class_order_data_index_tuple = []
    anomaly_weight_tuple = []
    for i in range(cluster_num):
        print("current calculate cluster  {0}  ...".format(i))
        tmp = np.where(cluster_assment == i)
        if tmp[0].size == 0:
            continue
        else:
            class_data = win_matrix[:, :, tmp[0]]
            cd_rows, cd_cols, cd_n = class_data.shape
            dictionary = class_data[:, int((win_size * win_size + 1)/2), :]
            dic_rows, dic_cols = dictionary.shape
            class_alpha = np.zeros((K, cd_cols, cd_n))
            class_index = np.zeros((K, cd_n))
            for j in range(cd_n):
                X = class_data[:, :, j]
                dictionary[:, (j * cd_cols): (j*cd_cols + cd_cols - 1)] = 0
                alpha, index, chosen_atom, resdiual = hyper.somp(dictionary, X, K)
                class_alpha[:, :, j] = alpha
                class_index[:, j] = index.transpose()
                resdiual_stack[:, :, save_num + j] = resdiual

            save_num = save_num + cd_n
            class_index = class_index.astype('int')
            class_global_alpha = np.zeros((dic_cols, cd_cols, cd_n))
            class_global_frequency = np.zeros((dic_cols, cd_cols, cd_n))
            for n_index in range(cd_n):
                class_global_alpha[class_index[:, n_index], :, n_index] = class_alpha[:, :, n_index]
                class_global_frequency[class_index[:, n_index], :, n_index] = 1

            posti_class_global_alpha = np.fabs(class_global_alpha)
            data_frequency = class_global_frequency[:, 0, :]
            frequency = np.sum(data_frequency, axis=1)
            sum_frequency = np.sum(frequency)
            norm_frequency = frequency/sum_frequency
            data_mean_alpha = np.mean(posti_class_global_alpha, axis=1)
            sum_alpha_2 = np.sum(data_mean_alpha, axis=1)
            norm_tmp = np.linalg.norm(sum_alpha_2)
            sparsity_score = sum_alpha_2 / norm_tmp
            anomaly_weight = norm_frequency
            anomaly_weight[frequency > 0] = sparsity_score[frequency > 0] / frequency[frequency > 0]
            # sparsity_score = sparsity_score * norm_frequency
            sparsity_sort_index = np.argsort(- sparsity_score)
            sparsity_sort_index = sparsity_sort_index.astype('int')
            frequency_sort_index = np.argsort(- norm_frequency)
            frequency_sort_index = frequency_sort_index.astype('int')
            tmp_class_dic_label = np.array(tmp[0])
            class_order_data_index_tuple.append(tmp_class_dic_label)
            selected_dic_num = np.round(selected_dic_percent * cd_n)
            selected_dic_num = selected_dic_num.astype('int')
            bg_dic_ac_tuple.append(tmp_class_dic_label[sparsity_sort_index[0: selected_dic_num]])
            bg_dic_fc_tuple.append(tmp_class_dic_label[frequency_sort_index[0: selected_dic_num]])
            anomaly_weight_tuple.append(anomaly_weight)
            bg_dic_tuple.append(dictionary[:, sparsity_sort_index[0: selected_dic_num]])

            # sio.savemat(result_path + "dic_{0}_frequency.mat".format(i), {'dic_frequency': frequency})
            # sio.savemat(result_path + "dic_{0}_reflect.mat".format(i), {'dic_reflect': sum_alpha_2})

    bg_dic = np.column_stack(bg_dic_tuple)
    bg_dic_ac_label = np.hstack(bg_dic_ac_tuple)
    bg_dic_fc_label = np.hstack(bg_dic_fc_tuple)
    anomaly_weight = np.hstack(anomaly_weight_tuple)
    class_order_data_index = np.hstack(class_order_data_index_tuple)
    norm_res = np.zeros((wm_n, win_size * win_size))
    for i in range(wm_n):
        norm_res[i, :] = np.linalg.norm(resdiual_stack[:, :, i], axis=0)
    mean_norm_res = np.mean(norm_res, axis=1) * anomaly_weight.transpose()
    anomaly_level = mean_norm_res/np.linalg.norm(mean_norm_res)
    tg_sort_index = np.argsort(- anomaly_level)
    tg_dic = data2d[:, class_order_data_index[tg_sort_index[0: target_dic_num]]]
    print ("successs!!")

    sio.savemat("bg_dic.mat", {'bg_dic': bg_dic})
    sio.savemat("bg_dic_ac_label.mat", {'bg_dic_ac_label': bg_dic_ac_label})
    sio.savemat("bg_dic_fc_label.mat", {'bg_dic_fc_label': bg_dic_fc_label})
    sio.savemat("tg_dic.mat", {'tg_dic': tg_dic})
    tg_dic_label = class_order_data_index[tg_sort_index[0:target_dic_num]]
    sio.savemat("tg_dic_label.mat", {'tg_dic_label': tg_dic_label})
    return data2d, bg_dic, tg_dic, bg_dic_ac_label, tg_dic_label
