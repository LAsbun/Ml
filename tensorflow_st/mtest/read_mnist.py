#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: read_mnist.py
@time: 18-1-3 下午10:16
@desc:
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

images, labels = load_mnist("/home/sws/captcha_platform/mtest/MNIST_data")
print(len(images), len(labels))
print(type(images[0]), labels[0])
# print(images[0].reshape(28, 28), labels[0])
for i in images[0].reshape(28, 28):
    print("".join(map(lambda x: str(x).ljust(4), i)))
    # print(i[0])