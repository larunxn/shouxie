# -*- coding: utf-8 -*-
"""
@Time: 2023/11/18 13:14
@Author: Changmeng Yang
"""

import os
import numpy as np
from collections import defaultdict
from model import Model


class Config:
    epoch = 2
    batch_size = 2
    max_len = 10
    data_path = os.path.join('data', 'train0.txt')


def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_text, all_label = [], []
    for item in all_data:
        data_tmp = item.split('\t')
        if len(data_tmp) != 2:
            continue
        text_item, label_item = data_tmp
        all_text.append(text_item)
        all_label.append(label_item)

    assert len(all_text) == len(all_label)
    return all_text, all_label


class MyDataset:
    # 存数据
    def __init__(self, all_text, all_label, config, word_2_index, label_2_index):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = config.batch_size
        self.word_2_index = word_2_index
        self.label_2_index = label_2_index

    def __iter__(self):
        return MyDataLoader(self)

    def __getitem__(self, item):   # 一条一条数据
        text = self.all_text[item][:config.max_len]
        label = self.all_label[item]

        text_idx = [self.word_2_index[w] for w in text]
        label_idx = self.label_2_index[label]

        text_idx_p = text_idx + [0] * (config.max_len - len(text_idx))
        return text_idx_p, label_idx


class MyDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration

        batch_data = [self.dataset[i] for i in range(self.cursor, min(len(self.dataset.all_text), self.cursor+self.dataset.batch_size))]
        text_idx, label_idx = zip(*batch_data)

        self.cursor += self.dataset.batch_size
        return np.array(text_idx), np.array(label_idx)    # 转成矩阵，有shape


def build_word_2_index(all_text):
    word_2_index = defaultdict(int)
    word_2_index['<PAD>'] = 0
    for text in all_text:
        for item in text:
            word_2_index[item] = word_2_index.get(item, len(word_2_index))
    return word_2_index


def build_label_2_index(all_label):
    return {k: num for num, k in enumerate(set(all_label))}


if __name__ == '__main__':
    config = Config()
    all_text, all_label = read_data(config.data_path)

    word_2_index = build_word_2_index(all_text)
    label_2_index = build_label_2_index(all_label)

    dataset = MyDataset(all_text, all_label, config, word_2_index, label_2_index)

    model = Model()

    for epoch in range(config.epoch):
        for batch_data in dataset:
            batch_text_idx, batch_label_idx = batch_data

            predict = model.forward(batch_text_idx)
            print(batch_text_idx, batch_label_idx, "预测：" + str(predict))
        print(" * " * 30)

    print(' ')

