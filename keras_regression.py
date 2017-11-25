from sklearn.metrics import mean_squared_error

from GradienBoosting import format_output
from read_data import x_train, y_train, x_test
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.activations import elu
from keras.optimizers import Nadam
from keras.losses import mean_squared_error as msr_loss
from keras.callbacks import EarlyStopping

n_features = len(x_train.columns)

input_l = Input(shape = [n_features])
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


es = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
model.fit(x_train.values, y_train.values, validation_split=0.2, epochs=40, callbacks=[es])

# make predictions for test data

y_test = model.predict(x_test.values)
y_test = format_output(y_test)
y_test.to_csv("submission/result_mlp_elu_early.csv")


