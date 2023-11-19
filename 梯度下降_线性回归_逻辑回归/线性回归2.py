# -*- coding: utf-8 -*-
# @Time: 2023/11/19 12:50
# @Author: Changmeng Yang


# ------------- 数据获取 ---------------
xs = [i for i in range(2000,2023)]
ys = [5000, 8000,12000,15000,14000,18000,20000,25000,26000,32000,40000,42000,46000,50000,51000,53000,53000,54000,57000,58000,59000,59900,60000]

# ------------- 数据处理 ---------------
# min_max 归一化
x_min = min(xs)
x_max = max(xs)

y_min = min(ys)
y_max = max(ys)

xs_normal = [(i - x_min)/(x_max - x_min) for i in xs]
ys_normal = [(i - y_min)/(y_max - y_min) for i in ys]

# ------------- 参数定义 ---------------
k = 1
b = 0
lr = 0.01
epoch = 1000
# ------------- 模型训练 ---------------
for e in range(epoch):
    for x, y in zip(xs_normal, ys_normal):   # batch=1
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
        if epoch % 100 == 0:
            print(loss)
# ------------- 推理 ---------------
while True:
    input_x = int(input("请输入年份："))
    input_x_normal = (input_x - x_min) / (x_max - x_min)
    p = k*(input_x_normal) + b
    pp = p * (y_max - y_min) + y_min
    print(f"{input_x}年的房价为：{pp}")