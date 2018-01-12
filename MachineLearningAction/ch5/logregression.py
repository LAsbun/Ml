#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: logregression.py
@time: 1/8/18 7:26 PM
@desc: logistic回归  随机梯度上升

目的:
    利用现有数据对分类边界建立回归公司,以此进行分类.

    分类:二分 跳跃 使用了sigmoid 算法

    优化算法: 使用了随机梯度上升算法



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

    return np.array(data_mat), np.array(label_mat)

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

        print(weights)

    return weights


def plot_best_fit(func):

    from matplotlib import pyplot as plt

    data_mat, label_mat = load_data()
    try:
        weights = func(data_mat, label_mat).getA()
    except:
        weights, _ = func(data_mat, label_mat, iter_num=100)
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

def stoc_grad_scent0(data_mat, class_labels, iter_num=10):
    """
        随机梯度上升

    :param data_mat:
    :param class_labels:
    :return:
    """

    import random

    data_mat = data_mat

    m, n = np.shape(data_mat)

    alpha = 0.01

    _tmp = []

    weights = np.ones(n)

    for j in range(iter_num):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.001
            range_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[range_index]*weights))
            weights = weights + alpha * (class_labels[range_index] - h) * data_mat[range_index]

            del(data_index[range_index])
        # print(weights)

            _tmp.append(weights)

    return weights, _tmp

def plot_x():

    """
        绘制特征参数变化曲线
    :return:
    """

    from matplotlib import pyplot as plt

    _, _tmp = stoc_grad_scent0(*load_data(), iter_num=100)

    a = np.array(_tmp)
    print(a.shape)
    # for i in enumerate(_tmp):

    fig = plt.figure()

    _, n = a.shape

    for i in range(n):
        label = "w{}".format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(a[:, i], label=label)
        ax.legend()

    plt.show()


if __name__ == '__main__':
    # plot_x()

    # print(grad_ascent(*load_data()))

    plot_best_fit(stoc_grad_scent0)