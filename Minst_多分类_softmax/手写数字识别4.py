# -*- coding: utf-8 -*-
# @Time: 2023/12/3 21:15
# @Author: Changmeng Yang

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
    sum_ex = np.sum(ex, axis=1, keepdims=True)

    result = ex/sum_ex
    return result


def make_onehot(labels,class_num):
    result = np.zeros((labels.shape[0],class_num))

    for idx,cls in enumerate(labels):
        result[idx][cls] = 1
    return result

class Dataset:
    def __init__(self, all_images, all_labels):
        self.all_images = all_images
        self.all_labels = all_labels

    def __getitem__(self, index):
        image = self.all_images[index]
        label = self.all_labels[index]
        return image, label

    def __len__(self):
        return len(self.all_images)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_images = []
        batch_labels = []
        for i in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                raise StopIteration
            data = self.dataset[self.cursor]
            batch_images.append(data[0])
            batch_labels.append(data[1])

            self.cursor += 1
        return np.array(batch_images), np.array(batch_labels)


if __name__ == "__main__":
    train_images = load_images(os.path.join("..","data","mnist","train-images.idx3-ubyte")) / 255   # 归一化：softmax  e
    train_labels = make_onehot(load_labels(os.path.join("..","data","mnist","train-labels.idx1-ubyte")), 10)

    dev_images = load_images(os.path.join("..", "data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("..", "data", "mnist", "t10k-labels.idx1-ubyte"))

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    batch_size = 10
    shuffle = False

    train_dataset = Dataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle)

    w = np.random.normal(0, 1, size=(784, 10))
    b = np.random.normal(0, 1, size=(1, 10))

    epoch = 10
    lr = 0.1

    for e in range(epoch):
        for batch_images, batch_labels in train_dataloader:
            pre = batch_images @ w + b

            p = softmax(pre)

            loss = -np.mean(batch_labels * np.log(p))

            G = (p - batch_labels) / batch_size

            delta_w = batch_images.T @ G
            delta_b = np.mean(G, axis=0, keepdims=True)

            w -= delta_w * lr
            b -= delta_b * lr

        right = 0
        for batch_images,batch_labels in dev_dataloader:
            pre = batch_images @ w + b
            pre_idx = np.argmax(pre, axis=-1)

            right += np.sum(pre_idx == batch_labels)
        acc = right / len(dev_dataset)
        print(f"acc: {acc:.3f}")




