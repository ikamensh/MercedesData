import pandas as pd

x_train_file = 'test_mlp_features0.csv'
x_train = pd.read_csv(x_train_file)

#Filter konstante Werte
print(x_train.head())
