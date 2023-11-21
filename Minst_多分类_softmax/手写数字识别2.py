# -*- coding: utf-8 -*-
# @Time: 2023/11/19 18:39
# @Author: Changmeng Yang


import numpy as np
import os
import struct
import matplotlib.pyplot as plt


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28,28)

def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex)

    result = ex/sum_ex
    return  result


def make_onehot(labels,class_num):
    result = np.zeros((labels.shape[0],class_num))

    for idx,cls in enumerate(labels):
        result[idx][cls] = 1
    return result


if __name__ == "__main__":
    train_images = load_images(os.path.join("..","data","mnist","train-images.idx3-ubyte")) / 255   # 归一化：softmax  e
    train_labels = make_onehot(load_labels(os.path.join("..","data","mnist","train-labels.idx1-ubyte")), 10)

    dev_images = load_images(os.path.join("..", "data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("..", "data", "mnist", "t10k-labels.idx1-ubyte"))

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    w = np.random.normal(0, 1, size=(784, 10))
    b = np.random.normal(0, 1, size=(1, 10))

    epoch = 10
    lr = 0.001

    for e in range(epoch):
        for idx in range(len(train_images)):
            image = train_images[idx: idx+1]    # [:] 保持维度不变
            label = train_labels[idx: idx+1]

            pre = image @ w + b
            p = softmax(pre)

            loss = - np.sum(label*np.log(p))    # 多元交叉熵

            G = p - label                       # 定义G为L对C的导数: A*B=C
            delta_w = image.T @ G
            delta_b = G

            w -= delta_w*lr
            b -= delta_b*lr
        # print(f"{loss:.3f}")

        right_num = 0
        for idx in range(len(dev_images)):
            image = dev_images[idx:idx+1]
            label = dev_labels[idx]

            pre = image @ w + b

            pre_idx = int(np.argmax(pre, axis=1)[0])

            right_num += int(pre_idx == label)

        acc = right_num / len(dev_labels)
        print(f"acc: {acc}")