import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def load():
    data = np.loadtxt('data/imports-85.data', delimiter=',', dtype=np.unicode)
    l = list(range(2, 9)) + [14, 15, 17]
    data[data == '?'] = 0
    for i in l:
        data[:, i], uniques = pd.factorize(data[:, i])
    data = data.astype(np.float)
    data = minmax(data)
    return data


def minmax(data):
    n = np.shape(data)[1]
    for i in range(n):
        mmin = min(data[:, i])
        mmax = max(data[:, i])
        data[:, i] = (data[:, i] - mmin) / (mmax - mmin)
    return data


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = np.cov(meanRemoved, rowvar=0)
    eig_val, eig_vector = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eig_val)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eig_vector[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print('len:', topNfeat)
    # print('max var:', eig_vector[:,eigValInd[-1]])
    print('max var:', eig_val[0])
    # print(sum(eig_val))
    # print(eig_val/sum(eig_val))
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = np.mat(load())
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat


def tp():
    arr = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
    d, r = pca(arr, 1)
    print(d)

# if __name__ == '__main__':
# main()
# tp()
