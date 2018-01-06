#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: knn_date.py
@time: 12/15/17 4:37 PM
@desc:

    所谓的knn（k近邻算法）：
    本质上是根据一定的已标记数据，待分类数据与已标记数据计算所有的距离，
    选其中距离最短的前n个，这前n个中分类出现最多的分类，就是待分类数据的分类

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

    # tile（A, B) 重复A, B次   a, (b, c)  在行上重复b次，在列上重复c次
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2

    sqDis = sqDiffMat.sum(axis=1)

    distince = sqDis ** 0.5

    # argsort 返回从小到大的数据下标 a = array([3, 1, 2]) a.argsort()==> [1, 2, 0]
    sortedDis = distince.argsort()

    classCount = {}

    for i in range(k):
        vateLabel = labels[sortedDis[i]]
        classCount[vateLabel] = classCount.get(vateLabel, 0) + 1
    # classCount 是统计各种分类出现的额次数，最后按照从大到小排序，返回出现频率最高的那个
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def datingClassTest():

    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    noreMat, ranges, minVals = autoNorm(datingDataMat)
    m = noreMat.shape[0]
    numTestVecs = int(m*hoRatio)
    print(m)
    errorCount = 0
    # 这里 取 十分之一的数据作为测试，每次与剩余的十分之九分类比较
    for i in range(numTestVecs):
        classifyResult = classify0(noreMat[i, :], noreMat[numTestVecs:m, :],\
                                   datingLabels[numTestVecs:m], 3)
        # print("classifyResult: {}. Label:{}".format(classifyResult, datingLabels[i]))
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

    print(_min)
    print(_max)

    _range = _max - _min

    normDataSet = zeros(shape=shape(dataSet))
    print(dataSet.shape)
    print(_range)
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


def img2vec(file_name):
    """
        将图片转化成向量
    :param file_name:
    :return: VEC
    """

    returnVec = zeros([1, 1024])

    with open(file_name, "r") as f:
        for i in range(32):
            _line = f.readline()
            for j in range(32):
                returnVec[0, 32*i+j] = int(_line[j])
    return returnVec

def handwritingClassTest():

    hwLabels = []

    trainFileList = os.listdir('trainingDigits')

    m = len(trainFileList)
    trainMat = zeros((m, 1024))
    i = 0
    for tf in trainFileList:
        hwLabels.append(tf.split('_')[0])
        trainMat[i, :] = img2vec("trainingDigits/{}".format(tf))
        i += 1

    testFileList = os.listdir('testDigits')

    error_count = 0
    tm = len(testFileList)

    for i in testFileList:
        _tmp_vec = img2vec("testDigits/{}".format(i))
        pre = classify0(_tmp_vec, trainMat, hwLabels, 3)
        _tmp_label = i.split('_')[0]
        print("预测: {}. 实际: {}".format(pre, _tmp_label))
        if(pre != _tmp_label): error_count += 1

    print("{} {}".format(error_count, tm))

if __name__ == '__main__':
    # file_name = "datingTestSet2.txt"
    # draw(file_name)
    # datingClassTest()
    cur_path = os.getcwd()
    print(img2vec('testDigits/0_0.txt')[0, 64:128])
    print(os.listdir("testDigits"))
    handwritingClassTest()