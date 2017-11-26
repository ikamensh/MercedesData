# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:37:26 2017

@author: Deike

Script to analyze a model reagarding feature importance
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
x_train_preprocessed = pd.read_csv('../features/x_train_preprocessed.csv',index_col='ID')

file_model = '../modelling/GradientBoosting_best_estimator_pickle.sav'
loaded_model = pickle.load(open(file_model, 'rb'))

#Feature importance
df_imp = pd.DataFrame(loaded_model.feature_importances_,index=x_train_preprocessed.columns)

df_imp.plot()
df_imp.columns = ['Importance']
df_imp_sort = df_imp.sort_values('Importance',ascending=False)
df_imp_sort.head(20).plot.bar()