#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: knn_date.py
@time: 12/15/17 4:37 PM
@desc:
"""
import os
import operator
from numpy import zeros, array, shape, tile

import matplotlib
import matplotlib.pyplot as plt

def file2matrix(file_name):
    """
        导入训练数据
    :param file_name:
    :return: 数据矩阵 returnMat 和对应的类别classLableVector
    """

    dir = os.path.abspath(__file__)
    print(dir)

    with open(file_name, "r") as f:
        lines = f.readlines()
        line_len = len(lines)

    returnMat = zeros((line_len, 3))

    classLabelVector = []

    index = 0

    for line in lines:
        line = line.strip()
        _tmp_list = line.split('\t')

        returnMat[index, :] = _tmp_list[0:3]

        classLabelVector.append(int(_tmp_list[-1]))

        index += 1

    return returnMat, classLabelVector

def classify0(inX, dataSet, labels, k):
    """
    k 近邻算法
    :param inX:  输入向量
    :param dataSet:
    :param labels:
    :param k: k
    :return:
    """

    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2

    sqDis = sqDiffMat.sum(axis=1)

    distince = sqDis ** 0.5

    sortedDis = distince.argsort()

    classCount = {}

    for i in range(k):
        vateLabel = labels[sortedDis[i]]
        classCount[vateLabel] = classCount.get(vateLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def datingClassTest():

    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    noreMat, ranges, minVals = autoNorm(datingDataMat)
    m = noreMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifyResult = classify0(noreMat[i, :], noreMat[numTestVecs:m, :],\
                                   datingLabels[numTestVecs:m], 3)
        print("classifyResult: {}. Label:{}".format(classifyResult, datingLabels[i]))
        if(classifyResult != datingLabels[i]): errorCount += 1
    print("{}:{}".format(errorCount, numTestVecs))

def autoNorm(dataSet):
    """
        归一化
    :param dataSet: 数据集
    :return: normDataSet, _range, _min
     公式 Y = (x-Xmin)/(Xmax-Xmin)  Y ~(0, 1)
    """

    _min = dataSet.min(0)
    _max = dataSet.max(0)

    _range = _max - _min

    normDataSet = zeros(shape=shape(dataSet))
    print(dataSet.shape)
    m = dataSet.shape[0]

    normDataSet = dataSet - tile(_min, (m, 1))

    normDataSet = normDataSet / tile(_range, (m, 1))

    return normDataSet, _range, _min

def draw(file_name):
    datingDataMat, datingLabels = file2matrix(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), array(datingLabels))
    plt.show()


if __name__ == '__main__':
    file_name = "datingTestSet2.txt"
    draw(file_name)