# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:32:32 2017

@author: Deike
"""

import pandas as pd

df = pd.read_csv('../data/train.csv',index_col='ID')

x_train = df.drop('y',axis=1)
y_train = df.y

for item in x_train:
    if type(x_train[item].iloc[0]) == str:
        x_train[item] = x_train[item].astype('category')
        x_train[item] = x_train[item].cat.codes

print(x_train)

