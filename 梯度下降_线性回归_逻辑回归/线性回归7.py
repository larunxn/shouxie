# -*- coding: utf-8 -*-
# @Time: 2023/11/19 14:18
# @Author: Changmeng Yang


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def feature_scaler(feature):
    f_scaler = MinMaxScaler()
    f_scaler.fit(feature)
    feature = f_scaler.transform(feature)
    return feature


if __name__ == '__main__':
    all_data = pd.read_csv('上海二手房价.csv')
    prices = all_data["房价（元/平米）"].values.reshape(-1,1)
    prices = feature_scaler(prices)

    floors = all_data["楼层"].values.reshape(-1, 1)
    floors = feature_scaler(floors)

    years = all_data["建成年份"].values.reshape(-1, 1)
    years = feature_scaler(years)

    features = np.stack([prices, floors, years], axis=1).squeeze()

    k = np.array([1, 1, 1], dtype=float).reshape(-1, 1)
    b = 0

    epoch = 10
    lr = 0.1

    for e in range(epoch):
        pre = features @ k + b
        loss = np.mean((pre - prices) ** 2)
        G = (pre - prices) / pre.shape[0]
        delta_k = features.T @ G
        k -= lr*delta_k
        print(loss)
    print("")