# -*- coding: utf-8 -*-
"""
@Time: 2023/11/18 16:55
@Author: Changmeng Yang
"""


class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return x+1

    def __call__(self, x):
        return self.forward(x)