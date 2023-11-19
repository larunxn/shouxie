# -*- coding: utf-8 -*-
# @Time: 2023/11/19 13:15
# @Author: Changmeng Yang

import random

# ------------- 数据获取 ---------------
xs1 = [i for i in range(2000,2023)]
xs2 = [random.randint(60,150) for i in range(2000,2023)]
ys = [5000, 8000,12000,15000,14000,18000,20000,25000,26000,32000,40000,42000,46000,50000,51000,53000,53000,54000,57000,58000,59000,59900,60000]

# ------------- 数据处理 ---------------
# min_max 归一化
x1_min = min(xs1)
x1_max = max(xs1)

x2_min = min(xs2)
x2_max = max(xs2)

y_min = min(ys)
y_max = max(ys)

xs1_normal = [(i - x1_min)/(x1_max - x1_min) for i in xs1]
xs2_normal = [(i - x1_min)/(x1_max - x1_min) for i in xs1]
ys_normal = [(i - y_min)/(y_max - y_min) for i in ys]

# ------------- 参数定义 ---------------
k = 1
b = 0
lr = 0.1
epoch = 1000
# ------------- 模型训练 ---------------
for e in range(epoch):
    for x1, x2, y in zip(xs1_normal, xs2_normal, ys_normal):   # batch=1
        # ------------- 推理预测 ---------------
        predict = k * (x1 + x2) + b
        # ------------- 计算损失 ---------------
        loss = (predict - y) ** 2
        # ------------- 计算梯度 ---------------
        delta_k = 2 * (predict - y) * (x1+x2)
        delta_b = 2 * (predict - y) * 1
        # ------------- 更新参数 ---------------
        k -= delta_k * lr
        b -= delta_b * lr
    if epoch % 100 == 0:
        print(loss)
# ------------- 推理 ---------------
while True:
    input_x1 = int(input("请输入年份："))
    input_x1_normal = (input_x1 - x1_min) / (x1_max - x1_min)
    input_x2 = int(input("请输入年份："))
    input_x2_normal = (input_x2 - x2_min) / (x2_max - x2_min)
    p = k*(input_x1_normal + input_x2_normal) + b
    pp = p * (y_max - y_min) + y_min
    print(f"{input_x}年的房价为：{pp}")