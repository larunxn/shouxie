# -*- coding: utf-8 -*-
# @Time: 2023/12/6 16:04
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

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.normal(0, 1, size=(in_features, out_features))
        self.bias = np.zeros((1, out_features))

    def forward(self, x):
        result = x @ self.weight + self.bias
        self.x = x
        return result

    def backward(self,G):
        delta_w = self.x.T @ G
        delta_b = np.sum(G,axis=0)

        self.weight -= lr * delta_w
        self.bias -= lr * delta_b

        delta_x = G @ self.weight.T

        return delta_x


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.result = sigmoid(x)
        return self.result

    def backward(self, G):
        return G * self.result * (1 - self.result)


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        return softmax(x)


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

    linear1 = Linear(784, 256)
    sigmoid_layer = Sigmoid()
    linear2 = Linear(256, 300)
    linear3 = Linear(300, 10)

    epoch = 100
    lr = 0.0001

    for e in range(epoch):
        for batch_images, batch_labels in train_dataloader:
            H1 = linear1.forward(batch_images)  # 第一层
            H1_S = sigmoid_layer.forward(H1)           # 第二层
            H2 = linear2.forward(H1_S)          # 第三层
            pre = linear3.forward(H2)          # 第四层

            p = softmax(pre)
            loss = - np.mean(batch_labels*np.log(p))

            G4 = (p - batch_labels) / batch_images.shape[0]  #  第四层 矩阵 C 位置的导数
            G3 = linear3.backward(G4)
            delta_H1_S = linear2.backward(G3)
            G1 = sigmoid_layer.backward(delta_H1_S)  #  第一层 矩阵 C 位置的导数
            linear1.backward(G1)

        right = 0
        for batch_images,batch_labels in dev_dataloader:

            H1 = linear1.forward(batch_images)  # 第一层
            H1_S = sigmoid(H1)           # 第二层
            H2 = linear2.forward(H1_S)          # 第三层
            pre = linear3.forward(H2)          # 第四层

            pre_idx = np.argmax(pre, axis = -1)

            right += np.sum(pre_idx == batch_labels)
        acc = right/len(dev_dataset)
        print(f"acc:{acc:.3f}")