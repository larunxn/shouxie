# -*- coding: utf-8 -*-
# @Time: 2023/11/19 12:26
# @Author: Changmeng Yang


# sqrt(3): x ** 2 = 3...  x=

epoch = 100
lr = 0.01
x = 5
y = 3

for e in range(epoch):
    predict = x ** 2
    loss = (predict - y) ** 2
    delta_x = 2 * (predict - y) * 2 * x
    x -= delta_x*lr
    print(loss)

