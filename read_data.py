from preprocessing.x_train_preparer import preprocess_x, extract_y
from utility.visualize import visualize_data

_corr_criteria = 0.05

x_train = preprocess_x("data/train.csv")
x_test = preprocess_x("data/test.csv")

x_train, y_train = extract_y(x_train)
print(x_train.tail())
#visualize_data(x_train, y_train)

from sklearn.model_selection import train_test_split

x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.25)

import pandas as pd
features_mlp_train = pd.read_csv("features/train_mlp_features0.csv")
features_mlp_train.index = x_train.index

features_mlp_test = pd.read_csv("features/test_mlp_features0.csv")
features_mlp_test.index = x_test.index

x_train_aug = pd.merge(x_train, features_mlp_train, left_index=True, right_index=True)
x_test_aug = pd.merge(x_test, features_mlp_test, left_index=True, right_index=True)
print('Column amount: {}'.format(len(x_train_aug.columns)))
#Filter for non zero standard deviation
x_train_aug_std = x_train_aug.std()
x_train_aug_std_fil = x_train_aug_std[x_train_aug_std != 0.0]
x_train_aug = x_train_aug.filter(x_train_aug_std_fil.index.values)
x_test_aug = x_test_aug.filter(x_train_aug_std_fil.index.values)
print('Column amount: {}'.format(len(x_train_aug.columns)))
#Filter for correlation to label
ds_corr = x_train_aug.corrwith(y_train)
ds_corr = abs(ds_corr)
#apply filter
ds_corr_fil = ds_corr[ds_corr > _corr_criteria]
x_train_aug = x_train_aug.filter(ds_corr_fil.index.values)
x_test_aug = x_test_aug.filter(ds_corr_fil.index.values)
print('Column amount: {}'.format(len(x_train_aug.columns)))