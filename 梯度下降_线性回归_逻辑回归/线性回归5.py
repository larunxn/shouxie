# -*- coding: utf-8 -*-
# @Time: 2023/11/19 13:40
# @Author: Changmeng Yang

import random
import numpy as np

# ------------- 数据获取 ---------------
xs1 = np.array([i for i in range(2000,2023)]).reshape(-1, 1)
xs2 = np.array([random.randint(60,150) for i in range(2000,2023)]).reshape(-1, 1)
ys = np.array([5000, 8000,12000,15000,14000,18000,20000,25000,26000,32000,40000,42000,46000,50000,51000,53000,53000,54000,57000,58000,59000,59900,60000])

# ------------- 数据处理 ---------------
# min_max 归一化
x1_min = min(xs1)
x1_max = max(xs1)

x2_min = min(xs2)
x2_max = max(xs2)

y_min = min(ys)
y_max = max(ys)

xs1_normal = (xs1 - x1_min)/(x1_max - x1_min)
xs2_normal = (xs2 - x2_min)/(x2_max - x2_min)
xs_normal = np.stack([xs1_normal, xs2_normal], axis=1).squeeze(axis=2)
ys_normal = ((ys - y_min)/(y_max - y_min)).reshape(-1,1)

# ------------- 参数定义 ---------------
k = np.array([1, 1], dtype=float).reshape(-1,1)
b = 0
lr = 0.1
epoch = 1000
# ------------- 模型训练 ---------------
for e in range(epoch):
    # ------------- 推理预测 ---------------
    predict = xs_normal @ k + b
    # ------------- 计算损失 ---------------
    loss = (predict - ys_normal) ** 2
    # ------------- 计算梯度 ---------------
    G = delta_C = predict-ys_normal
    delta_k = xs_normal.T @ G
    # ------------- 更新参数 ---------------
    k -= lr * delta_k
    if epoch % 100 == 0:
        print(loss)
# ------------- 推理 ---------------
while True:
    input_x1 = int(input("请输入年份："))
    input_x1_normal = (input_x1 - x1_min) / (x1_max - x1_min)

    input_x2 = int(input("请输入大小："))
    input_x2_normal = (input_x2 - x2_min) / (x2_max - x2_min)

    # p = k*(input_x1_normal + input_x2_normal) + b
    p = np.array([input_x1_normal, input_x2_normal]).reshape(1, 2) @ k + b

    pp = p * (y_max - y_min) + y_min
    print(f"房价为：{pp}")
