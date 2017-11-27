from datetime import time
from GradienBoosting import format_output
from read_data import x_train, y_train, x_test
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.activations import elu
from keras.optimizers import Nadam
from keras.losses import mean_squared_error as msr_loss
from keras.callbacks import EarlyStopping
import tensorflow as tf

n_features = len(x_train.columns)
input_l = Input(shape = [n_features])
print(input_l)
hidden0 = Dense(512, activation=elu) (input_l)
drop = Dropout(0.3) (hidden0)
hidden1 = Dense(128, activation=elu) (drop)
drop = Dropout(0.3) (hidden1)
hidden2= Dense(32) (drop)
dec1 = Dense(128, activation=elu) (hidden2)
dec2 = Dense(128, activation=elu) (dec1)
output = Dense(n_features) (dec2)


model = Model(input_l, output)
model.compile(Nadam(lr=2e-4), msr_loss)

features_model = Model(input_l, hidden2)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
model.fit(x_test.values, x_test.values, validation_split=0.2, epochs=45, callbacks=[es])

# make predictions for test data

import pandas as pd

last_layer_feats = pd.DataFrame(features_model.predict(x_train.values))
last_layer_feats.to_csv("features/train_autoenc_a_features1.csv")

last_layer_feats = pd.DataFrame(features_model.predict(x_test.values))
last_layer_feats.to_csv("features/test_autoenc_a_features1.csv")


writer = tf.summary.FileWriter('./my_graph', tf.get_default_graph())
writer.close()
