# coding=utf-8

import numpy as np
from scipy import linalg


def LRSR(DictLRR, DictSRC, data, beta, lmda):
    """
    
    :param DictLRR: the background dictionary 
    :param DictSRC: the anomaly dictionary
    :param data: the normalized data
    :param beta: parameters
    :param lmda: parameters 
    :return: Z: the low rank coefficeinets
             S: the sparse coefficients
             E: the noise
    """

    dataRows, dataCols = data.shape
    DLRows, DLCols = DictLRR.shape
    DSRows, DSCols = DictSRC.shape
    ILRR = np.eye(DLCols)
    ISRC = np.eye(DSCols)
    Z = np.zeros((DLCols, dataCols))
    J = np.zeros((DLCols, dataCols))
    E = np.zeros((dataRows, dataCols))
    S = np.zeros((DSCols, dataCols))
    L = np.zeros((DSCols, dataCols))
    Y1 = np.zeros((dataRows, dataCols))
    Y2 = np.zeros((DLCols, dataCols))
    Y3 = np.zeros((DSCols, dataCols))
    mu = 0.0001
    mu_max = 10 ** 10
    p = 1.1
    err = 0.000001
    itera = 1
    inv_Z = np.linalg.inv(np.dot(DictLRR.transpose(), DictLRR) + ILRR)
    inv_S = np.linalg.inv(np.dot(DictSRC.transpose(), DictSRC) + ISRC)

    while itera < 500:
        print("iteration:{0}".format(itera))
        # update J
        operator1 = 1 / mu
        tmpJ = Z + Y2 / mu
        Ju, Jsigma, Jvt = linalg.svd(tmpJ, full_matrices=False)
        # threshold1 =1/mu
        evp = Jsigma[Jsigma > operator1].shape[0]
        if evp >= 1:
            Jsigma[0:evp] -= operator1
            JsigmaM = np.diag(Jsigma[0:evp])
            print ("current evp is: {0}".format(evp))
        else:
            evp = 1
            JsigmaM = 0

        J = np.dot(np.dot(Ju[:, 0:evp], JsigmaM), Jvt[0:evp, :])
        # update E
        operator3 = lmda / mu
        tmpE = data - np.dot(DictLRR, Z) - np.dot(DictSRC, S) + Y1 / mu
        terows, tecols = tmpE.shape
        for i in range(tecols):
            tmpValue1 = linalg.norm(tmpE[:, i])
            if tmpValue1 > operator3:
                E[:, i] = ((tmpValue1 - operator3) / tmpValue1) * tmpE[:, i]
            else:
                E[:, i] = 0
        # update L
        tmpL = S + Y3 / mu
        operator2 = beta / mu
        tmpL[tmpL > operator2] -= operator2
        tmpL[tmpL < -operator2] += operator2
        tmpL[(tmpL >= -operator2) & (tmpL <= operator2)] = 0
        L = tmpL.copy()
        # update Z
        tmpZ = np.dot(DictLRR.transpose(), data - np.dot(DictSRC, S) - E) + J + \
            (np.dot(DictLRR.transpose(), Y1) - Y2) / mu
        Z = np.dot(inv_Z, tmpZ)
        # update S
        tmpS = np.dot(DictSRC.transpose(), data - np.dot(DictLRR, Z) - E) + L + \
            (np.dot(DictSRC.transpose(), Y1) - Y3) / mu
        S = np.dot(inv_S, tmpS)
        # update Y1,Y2,Y3
        T1 = data - np.dot(DictLRR, Z) - E - np.dot(DictSRC, S)
        T2 = Z - J
        T3 = S - L
        Y1 += mu * T1
        Y2 += mu * T2
        Y3 += mu * T3
        # update mu
        err1 = linalg.norm(T1, np.inf)
        err2 = linalg.norm(T2, np.inf)
        err3 = linalg.norm(T3, np.inf)
        rlerr = max(err1, err2, err3)
        mu = min(p * mu, mu_max)

        itera += 1
        print("max err is:{0}".format(rlerr))
        print("current mu is:{0}".format(mu))
        if rlerr < err:
            break
    return Z, E, S
