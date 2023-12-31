# -*- coding: utf-8 -*-
# @Time: 2023/11/19 12:51
# @Author: Changmeng Yang

# y = kx + b

# ------------- 数据获取 ---------------
xs = [i for i in range(2000,2023)]
ys = [5000, 8000,12000,15000,14000,18000,20000,25000,26000,32000,40000,42000,46000,50000,51000,53000,53000,54000,57000,58000,59000,59900,60000]

# ------------- 数据处理 ---------------

# ------------- 参数定义 ---------------
k = 1
b = 0
lr = 0.01
epoch = 20
# ------------- 模型训练 ---------------
for e in range(epoch):
    for x, y in zip(xs, ys):   # batch=1
        # ------------- 推理预测 ---------------
        predict = k * x + b
        # ------------- 计算损失 ---------------
        loss = (predict - y) ** 2
        # ------------- 计算梯度 ---------------
        delta_k = 2 * (predict - y) * x
        delta_b = 2 * (predict - y) * 1
        # ------------- 更新参数 ---------------
        k = k - delta_k * lr
        b -= delta_b * lr
# ------------- 推理 ---------------
input_x = input("请输入年份：")
p = k*(int(input_x)) + b
print(f"{input_x}年的房价为：{p}")





