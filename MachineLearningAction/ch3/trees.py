#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: trees.py
@time: 18-1-6 下午1:36
@desc: 决策树

ID3  根据熵，信息熵增益公式。 熵越大表示越混乱，每次按照熵减小的方向分数据集
最后生成决策树。

"""




from math import log
from operator import itemgetter

def calcShannonEnt(dataSet):
    """
        计算信息熵  熵越高表示混乱程度就越大
    :param dataSet:
    :return:
    """

    m_len = len(dataSet)

    label_dic = {}

    for ds in dataSet:

        _class = ds[-1]

        if _class not in label_dic.keys():
            label_dic[_class] = 1
        else:
            label_dic[_class] += 1

    shanoEn = 0.0
    for key in label_dic:
        prob = label_dic[key] / m_len
        shanoEn += - prob * log(prob, 2)

    return shanoEn

def splitDataSet(dataSet, axis, value):
    """
        根据特征以及特征值，返回分类
    :param dataSet:  待分类数据集
    :param axis: 特征
    :param value: 特征值
    :return: 分好的数据集
    """

    retDataSet = []

    for ds in dataSet:
        if ds[axis] == value:
            _tmp = ds[:axis]
            _tmp.extend(ds[axis+1:])
            retDataSet.append(_tmp)

    return retDataSet

def selectBestFeatureToSpilit(dataSet):
    """
        选择最好的数据集分类特征
        信息增益公式： http://blog.csdn.net/lemon_tree12138/article/details/51837983
        IG(S|T)=Entropy(S)−∑value(T)|Sv|SEntropy(Sv)
    :param dataSet:
    :return:
    """

    feature_list = len(dataSet[0]) - 1

    base_en = calcShannonEnt(dataSet)

    best_fea = -1

    shan_ig = 0

    for i in range(feature_list):

        _tmp_feature_values = [e[i] for e in dataSet]

        _tmp_feature_values_set = set(_tmp_feature_values)

        _tmp_i_sh = 0

        for fv in _tmp_feature_values_set:
            # 某个特征 特征值
            _tmp_feature_value_split_list = splitDataSet(dataSet, i, fv)

            _prob = len(_tmp_feature_value_split_list) / len(dataSet)

            _tmp_i_sh += _prob * calcShannonEnt(_tmp_feature_value_split_list)

        _tmp_i_ig = base_en - _tmp_i_sh

        print("{} {}".format(i, _tmp_i_ig))

        if _tmp_i_ig > shan_ig:
            best_fea = i
            shan_ig = _tmp_i_ig

    return best_fea

def majorityCnt(classList):
    """
        多数表绝
    :param classList: []
    :return:
    """

    _tmp_dic = {}

    for cl in classList:

        if cl not in _tmp_dic.keys():
            _tmp_dic = 1

        else:
            _tmp_dic[cl] += 1

    sortedClassDict = sorted(_tmp_dic.items(),\
                             key=itemgetter(1), reverse=True)

    return sortedClassDict[0][0]

def createTree(dataSet, labels):
    """
        返回决策树
    :param dataSet: 数据集
    :param labels: 标记
    :return:
    """
    classList = [e[-1] for e in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 如果数据集只有一种结果
        return classList[0]

    if len(dataSet[0]) == 1:
        # 如果分类到最后一个节点还没有分出来, 投票表决
        return majorityCnt(classList)

    bestFeature = selectBestFeatureToSpilit(dataSet)

    print(bestFeature, labels)

    bestFeatureLabel = labels[bestFeature]

    myTree = {bestFeatureLabel: {}}

    del labels[bestFeature]

    feaValues = [e[bestFeature] for e in dataSet]

    feaValuesSet = set(feaValues)

    for va in feaValuesSet:
        subLabels = labels[:]
        myTree[bestFeatureLabel][va] = createTree(splitDataSet(dataSet, bestFeature, va), subLabels)

    return myTree

def classfy(inputTree, featureLabels, testVec):
    """
        使用决策树进行分类
    :param inputTree: 已经分好的决策树
    :param featureLabels: 决策树特征
    :param testVec: 测试数据
    :return:
    """

    firstStr = list(inputTree.keys())[0]
    secDict = inputTree[firstStr]
    featIndex = featureLabels.index(firstStr)
    for k in secDict.keys():
        if testVec[featIndex] == k:
            if type(secDict[k]).__name__ == "dict":
                classLabel = classfy(secDict[k], featureLabels, testVec)
            else:
                classLabel = secDict[k]

    return classLabel