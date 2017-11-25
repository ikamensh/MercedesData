# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:32:32 2017

@author: Deike
"""

import pandas as pd

def preprocess_x(filepath):
    x_train = pd.read_csv(filepath, index_col='ID')

    for item in x_train:
        if type(x_train[item].iloc[0]) == str:
            x_train[item] = x_train[item].astype('category')
            x_train[item] = x_train[item].cat.codes

    return x_train

def extract_y(x_train):
    y_train = x_train.y
    x_train = x_train.drop('y', axis=1)

    return x_train, y_train


