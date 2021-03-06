from catboost import CatBoostRegressor

from GradienBoosting import format_output
from read_data import y_train
from read_data import x_train_auto, x_test_auto
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import from_model
from utility.mean_squared_error import compute_mean_squared_error

x_train, x_val, y_train, y_val = train_test_split(x_train_auto, y_train, test_size=0.25)



model = CatBoostRegressor(depth=5,
                      learning_rate=0.1,
                      loss_function='RMSE',
                      iterations=200, eval_metric="RMSE")

print("###### model.get_params()", model.get_params())

eval_set = (x_val, y_val)
model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

# make predictions for test data

y_pred = model.predict(x_val)

print("--- mean_squared_error ", compute_mean_squared_error(y_pred, y_val))

y_test = model.predict(x_test_auto)
y_test = format_output(y_test)
y_test.to_csv("submission/result_catboost_autoenc.csv")