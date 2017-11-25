from sklearn.metrics import mean_squared_error

from GradienBoosting import format_output
from read_data import x_train_split, x_val, y_train_split, y_val, x_test
from keras.models import Model
from keras.layers import Dense, Input
from keras.activations import elu
from keras.optimizers import Nadam
from keras.losses import mean_squared_error as msr_loss

n_features = len(x_train_split.columns)

input_l = Input(shape = [n_features])
hidden1 = Dense(128, activation=elu) (input_l)
hidden2 = Dense(128, activation=elu) (hidden1)
hidden3= Dense(32, activation=elu) (hidden2)
output = Dense(1) (hidden3)

model = Model(input_l, output)
model.compile(Nadam(lr=5e-4), msr_loss)


eval_set = (x_val, y_val)
model.fit(x_train_split.values, y_train_split.values, epochs=20)

# make predictions for test data

y_pred = model.predict(x_val.values)

print(mean_squared_error(y_val.values,y_pred))

y_test = model.predict(x_test.values)
y_test = format_output(y_test)
y_test.to_csv("submission/result_mlp_elu.csv")


