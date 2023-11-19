# -*- coding: utf-8 -*-
# @Time: 2023/11/19 12:37
# @Author: Changmeng Yang

import numpy as np
# (x1-3)**2 + (x2+4)**2 =  0

x1, x2 = 5, -6
label = 0
lr = 0.001
epoch = 1000

for e in range(epoch):
    predict = (x1 - 3) ** 2 + (x2 + 4) ** 2
    loss = (predict - label) ** 2
    delta_x1 = 2 * (predict - label) * 2 * (x1-3)
    delta_x2 = 2 * (predict - label) * 2 * (x2+4)

    x1 -= lr * delta_x1
    x2 -= lr * delta_x2

    if e % 100 == 0:
        print(loss)