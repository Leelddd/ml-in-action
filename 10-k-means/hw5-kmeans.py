import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import *
from itertools import cycle


def load_dataset():
    d = pd.read_csv('data/Sales_Transactions_Dataset_Weekly.csv', dtype=unicode).values
    # delete P + 52 week + min max
    d = np.delete(d, list(range(55)), axis=1)
    return d.astype(float)


# calculate euler distance
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


# def distEclud(x, y):
#     return 1 - dot(x.A[0], y.A[0]) / (linalg.norm(x.A[0]) * linalg.norm(y.A[0]))
# return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(data, k):
    """random select k centroid
    :param data: data matrix
    :param k: k cluster
    :return: centroids - kxn k centroids, each with n dimension
    """
    n = shape(data)[1]
    centroids = mat(zeros((k, n)))
    # create random cluster centers, within bounds of each dimension
    for j in range(n):
        minJ = min(data[:, j])
        rangeJ = float(max(data[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(data, k, dist=distEclud, createCent=randCent):
    """basic k means algorithms
    :param data: data matrix
    :param k: k cluster
    :param dist: distance function
    :param createCent: random centroid create function
    :return: centroids and clusterAssment(line_id to cluster_id map)
    """
    m = np.shape(data)[0]
    # <line_id, cluster_id, distance> dict
    clusterAssment = mat(zeros((m, 2)))
    # random create k cluster
    centroids = createCent(data, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # for each data point assign it to the closest centroid
        for i in range(m):
            minDist = inf
            minIndex = -1
            # for each centroid, calculate distance between centroid and data[i]
            for j in range(k):
                distJI = dist(centroids[j], data[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):  # recalculate centroids
            ptsInClust = data[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],
                               :]  # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment


def draw(cluster_id, data):
    print('cluster %d has %d transaction data' % (cluster_id, len(data)))
    plt.figure(cluster_id)
    x = list(range(52))
    plt.title('k-means-%d' % cluster_id)
    # cycol = cycle('bgrcmykw')
    for i in range(len(data)):
        # plt.plot(x, data[i], c=next(cycol))
        plt.plot(x, data[i], c='blue')
    # plt.legend()
    plt.savefig('out/%d.png' % cluster_id)


def test(data, filepath: str):
    d = loadtxt(filepath)

    cluster_cnt = int(max(d[:, 0])) + 1

    for i in range(cluster_cnt):
        cnt = len(data[d[:, 0] == i])
        if cnt > 0:
            draw(i, data[d[:, 0] == i])


def train(dataMat, method='basic'):
    if method == 'basic':
        centList, myNewAssments = kMeans(dataMat, 100)
    else:
        centList, myNewAssments = biKmeans(dataMat, 100)

    m = mean([distEclud(dataMat[i], dataMat[j]) for i in range(800) for j in range(800)])
    print('average distance before cluster is %f' % m)
    print('useing %s method, average distance after cluster is %f' % (method, mean(myNewAssments.A[:, 1])))

    savetxt('out/%s_cluster.txt' % method, myNewAssments.A)


if __name__ == '__main__':
    data = load_dataset()
    # train(np.mat(data))
    # train(np.mat(data), 'binary')
    # test(data, 'out/basic_cluster.txt')
    test(data, 'out/binary_cluster.txt')
