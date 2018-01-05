#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: mtest.py
@time: 12/20/17 2:09 PM
@desc:
"""
from math import exp
import numpy as np
import matplotlib.pyplot as plt

distance = 156

def sigmoid(x):
    return (1 / (1 + exp((-x+5)))) * distance

sli = 13

x = np.linspace(1, 8, sli)

print(sigmoid(1))

# y = (1 / (1 + exp(-x + 3))) * distance
# y = (1 / (1 + exp((-x) + 3))) * distance
#
plt.figure()
_tmp_list = [int(sigmoid(i)) for i in range(sli)]

count = sli - 1
while count > 0:
    _tmp_list[count] = _tmp_list[count] - _tmp_list[count-1]
    count -= 1


plt.plot(x, _tmp_list)
print(_tmp_list)
plt.show()