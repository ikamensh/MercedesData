from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from GradienBoosting import format_output
from read_data import x_train_split, x_val, y_train_split, y_val, x_test

model = CatBoostRegressor(depth=5,
                      learning_rate=0.1,
                      loss_function='RMSE',
                      iterations=200, eval_metric="RMSE")

print(model.get_params().keys())

eval_set = (x_val, y_val)
model.fit(x_train_split, y_train_split, eval_set=eval_set, verbose=False)

# make predictions for test data

y_pred = model.predict(x_val)
print(mean_squared_error(y_val,y_pred))

y_test = model.predict(x_test)
y_test = format_output(y_test)
y_test.to_csv("submission/result_catboost.csv")


