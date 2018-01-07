#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: bayes.py
@time: 18-1-7 下午3:38
@desc: 朴素贝叶斯

    假设所有特征的权重都是一样

    分为两种： 贝努利（只考虑出不出现），多项式（考虑出现次数）

    注意：
      1 计算概率的时候可能会有0次的出现，为了避免这种情况的出现， 分子初始化为1，分母初始化为2
      2 会出现下溢出（多个很小的数相乘～0），使用对数  ln(a×b) = ln(a) + ln(b)
"""
from numpy import *


def loadDataSet():
    """
        返回数据集
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
        创建词汇表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
        将文本中的单词对应的
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    """
        将文本中的单词对应的
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB(trainMat, trainCate):
    """

    :param trainMat: 训练集
    :param trainCate: 分类集
    :return: P(ci), p(W|ci)  因为ci只有0或1 所以返回的是p(1) p(W|1) p(W|0)
    """

    # P1 发生的概率
    P1 = sum(trainCate) / len(trainMat)

    p1Num = ones(len(trainMat[0]))
    p0Num = ones(len(trainMat[0]))
    p1Sum = p0Sum = 2

    for i in range(len(trainMat)):

        if trainCate[i] == 1:
            p1Num += trainMat[i]
            p1Sum += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Sum += sum(trainMat[i])


    return P1, log(p1Num/p1Sum), log(p0Num/p0Sum)


def classfyNB(vec2classfy, p0vec, p1vec, pClass1):
    """
        计算待测数据各类概率，返回最大类
    :param vec2classfy: 待测向量
    :param p0vec:
    :param p1vec:
    :param pClass1:
    :return:
    """

    p0 = sum(vec2classfy * p0vec) + log(1 - pClass1)
    p1 = sum(vec2classfy * p1vec) + log(pClass1)

    # print(p0)

    if p0 > p1:
        return 0
    else:
        return 1

def testingNB():

    listPosts, listClasses = loadDataSet()

    myV = createVocabList(listPosts)

    # 训练集array
    _tmp_vec = []

    for lp in listPosts:
        _tmp_vec.append(setOfWords2Vec(myV, lp))

    p1, pw1, pw0 = trainNB(array(_tmp_vec), array(listClasses))

    testE = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myV, testE))

    print("{}: {}".format(testE, classfyNB(thisDoc, pw0, pw1, p1)))

    testE = ["stupid", "garbage"]

    thisDoc = array(setOfWords2Vec(myV, testE))

    print("{}: {}".format(testE, classfyNB(thisDoc, pw0, pw1, p1)))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        try:
            wordList = textParse(open('email/ham/%d.txt' % i).read())
        except Exception as e:
            print(i)
            raise e
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pSpam, p1V, p0V = trainNB(array(trainMat), array(trainClasses))

    errorCount = 0

    # print(testSet)
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # print(wordVector)
        if classfyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            print('error :', docIndex, classfyNB(array(wordVector), p0V, p1V, pSpam), classList[docIndex])
            errorCount += 1
    print(
    'the error rate is : ', float(errorCount) / len(testSet))


if __name__ == '__main__':
    spamTest()