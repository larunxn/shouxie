# -*- coding: utf-8 -*-
# @Time: 2023/12/6 15:23
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


def make_onehot(labels,class_num):
    result = np.zeros((labels.shape[0],class_num))

    for idx,cls in enumerate(labels):
        result[idx][cls] = 1
    return result


class Dataset:
    def __init__(self,all_images,all_labels):
        self.all_images = all_images
        self.all_labels = all_labels

    def __getitem__(self, index):
        image = self.all_images[index]
        label = self.all_labels[index]

        return image,label

    def __len__(self):
        return len(self.all_images)


class DataLoader:
    def __init__(self,dataset,batch_size,shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        batch_imges = []
        batch_labels = []
        for i in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                break

            data = self.dataset[self.cursor]

            batch_imges.append(data[0])
            batch_labels.append(data[1])

            self.cursor += 1
        return np.array(batch_imges),np.array(batch_labels)


def softmax(x):
    x = np.clip(x,-1e10,100)
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)

    result = ex/sum_ex

    result = np.clip(result,1e-10,1e10)
    return  result

def sigmoid(x):
    x = np.clip(x,-100,1e10)
    result = 1/(1 + np.exp(-x))
    return result

if __name__ == "__main__":
    train_images = load_images(os.path.join("..","data","mnist","train-images.idx3-ubyte"))/255
    train_labels = make_onehot(load_labels(os.path.join("..","data","mnist","train-labels.idx1-ubyte")),10)

    dev_images = load_images(os.path.join("..", "data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("..", "data", "mnist", "t10k-labels.idx1-ubyte"))

    train_images = train_images.reshape(60000,784)
    dev_images = dev_images.reshape(-1,784)

    batch_size = 50
    shuffle = False

    train_dataset = Dataset(train_images,train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle)

    w1 = np.random.normal(0,1,size=(784,256))
    w2 = np.random.normal(0,1,size=(256,300))
    w3 = np.random.normal(0,1,size=(300,10))

    b1 = np.zeros((1,256))
    b2 = np.zeros((1,300))
    b3 = np.zeros((1,10))

    epoch = 100
    lr = 0.0001

    for e in range(epoch):
        for batch_images,batch_labels in train_dataloader:
            H1 = batch_images @ w1 + b1  # 第一层
            H1_S = sigmoid(H1)           # 第二层
            H2 = H1_S @ w2 + b2          # 第三层
            pre = H2 @  w3 + b3          # 第四层

            p = softmax(pre)
            loss = - np.mean(batch_labels*np.log(p))

            G4 = (p - batch_labels) / batch_images.shape[0]  #  第四层 矩阵 C 位置的导数

            delta_w3 = H2.T @ G4
            G3 = delta_H2 = G4 @ w3.T   # 第三层 矩阵 C 位置的导数

            delta_w2 = H1_S.T @ G3
            delta_H1_S = G3 @ w2.T      # 第二层 矩阵 C 位置的导数

            G1 = delta_H1 = delta_H1_S * ( H1_S * (1-H1_S))  #  第一层 矩阵 C 位置的导数
            delta_w1 = batch_images.T  @ G1


            delta_b3 = np.mean(G4,axis=0,keepdims=True)
            delta_b2 = np.mean(G3,axis=0,keepdims=True)
            delta_b1 = np.mean(G1,axis=0,keepdims=True)

            w1 -= lr * delta_w1
            w2 -= lr * delta_w2
            w3 -= lr * delta_w3
            b1 -= lr * delta_b1
            b2 -= lr * delta_b2
            b3 -= lr * delta_b3

        right = 0
        for batch_images,batch_labels in dev_dataloader:

            H1 = batch_images @ w1 + b1  # 第一层
            H1_S = sigmoid(H1)  # 第二层
            H2 = H1_S @ w2 + b2  # 第三层
            pre = H2 @ w3 + b3  # 第四层

            pre_idx = np.argmax(pre,axis = -1)

            right += np.sum(pre_idx == batch_labels)
        acc = right/len(dev_dataset)
        print(f"acc:{acc:.3f}")