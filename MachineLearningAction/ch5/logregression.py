#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: logregression.py
@time: 1/8/18 7:26 PM
@desc: logistic回归
"""

import numpy as np

def load_data():

    data_mat = []
    label_mat = []

    with open('testSet.txt') as f:
        for i in f:
            nums = list(map(float, i.split()))
            data_mat.append([1.0, nums[0], nums[1]])
            label_mat.append(int(nums[2]))

    return data_mat, label_mat

def sigmoid(inx):
    """

    :param inx: vector
    :return:
    """

    return 1.0 / (1 + np.exp(-inx))

def grad_ascent(data_mat, lables_mat):
    """

    :param data_mat:
    :param lables_mat:
    :return:
    """

    data_mat = np.mat(data_mat)

    lables_mat = np.mat(lables_mat).transpose()

    m, n = data_mat.shape

    alpha = 0.001

    max_cycle = 500

    weights = np.ones((n, 1))

    for k in range(max_cycle):
        h = sigmoid(data_mat * weights)
        error = (lables_mat - h)
        weights = weights + alpha * data_mat.transpose() * error

    return weights


def plot_best_fit():

    from matplotlib import pyplot as plt

    data_mat, label_mat = load_data()

    weights = grad_ascent(data_mat, label_mat).getA()
    
    x1 = [] 
    x2 = []
    
    y1 = []
    y2 = []
    
    for i, k in enumerate(data_mat):
        
        if label_mat[i] == 1:
            x1.append(k[1])
            y1.append(k[2])
        else:
            x2.append(k[1])
            y2.append(k[2])

    flg = plt.figure()
    ax = flg.add_subplot(111)
    
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)

    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)

    plt.show()
x



if __name__ == '__main__':

    # print(grad_ascent(*load_data()))

    plot_best_fit()