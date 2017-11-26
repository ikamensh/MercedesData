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
hidden2 = Dense(128, activation=elu) (drop)
drop = Dropout(0.3) (hidden2)
hidden3= Dense(32, activation=elu) (drop)
output = Dense(1) (hidden3)

model = Model(input_l, output)
model.compile(Nadam(lr=2e-4), msr_loss)

features_model = Model(input_l, hidden3)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
model.fit(x_train.values, y_train.values, validation_split=0.2, epochs=45, callbacks=[es])

# make predictions for test data

y_test = model.predict(x_test.values)
y_test = format_output(y_test)
y_test.to_csv("submission/result_mlp_elu_early.csv")

# import tensorflow as tf
# sess = tf.Session()
# last_layer_feats = sess.run(hidden3, feed_dict={input_l : x_test.values})
import pandas as pd

last_layer_feats = pd.DataFrame(features_model.predict(x_train.values))
last_layer_feats.to_csv("features/train_mlp_features1.csv")

last_layer_feats = pd.DataFrame(features_model.predict(x_test.values))
last_layer_feats.to_csv("features/test_mlp_features1.csv")


writer = tf.summary.FileWriter('./my_graph', tf.get_default_graph())
writer.close()
