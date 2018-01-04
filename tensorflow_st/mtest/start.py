#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: start.py
@time: 18-1-3 下午9:58
@desc:
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)   # 分类模型


# 下面的是分类精度

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 找到一个合适的优化算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化全局参数
tf.global_variables_initializer().run()

# 训练1000次，每次取100个
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
# 预测
cor_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

acc = tf.reduce_mean(tf.cast(cor_prediction, tf.float32))

print(acc.eval({x: mnist.test.images, y_: mnist.test.labels}))
