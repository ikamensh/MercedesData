from sklearn.metrics import mean_squared_error

from GradienBoosting import format_output
from read_data import x_train_split, x_val, y_train_split, y_val, x_test, x_train_aug, x_test_aug



from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


model = XGBRegressor(max_depth=5,
                      learning_rate=0.02,
                      objective='reg:linear',
                      n_estimators=300,
                     booster="gblinear")

print(model.get_params().keys())

eval_set = [(x_val, y_val)]
model.fit(x_train_split, y_train_split, eval_metric="rmse", eval_set=eval_set, verbose=True)

print(model.feature_importances_)

y_pred = model.predict(x_val)
print(mean_squared_error(y_val,y_pred))

y_test = model.predict(x_test)
y_test = format_output(y_test)
y_test.to_csv("submission/result_xgboost.csv")


