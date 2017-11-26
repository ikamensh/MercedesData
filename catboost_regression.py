from catboost import CatBoostRegressor

from GradienBoosting import format_output
from read_data import x_train_split, x_val, y_train_split, y_val, x_test
from utility.mean_squared_error import compute_mean_squared_error
from sklearn.model_selection import GridSearchCV


search_param = {'depth': [3,4,5,6,7],
                'learning_rate': [0.02,0.05,0.1,0.15,0.3],
                'iterations': [100,200,300,400,500]}

gridsearch = GridSearchCV(estimator = CatBoostRegressor(depth=5,
                                                          learning_rate=0.1,
                                                          loss_function='RMSE',
                                                          iterations=200, eval_metric="RMSE"),
                          param_grid = search_param)




eval_set = (x_val, y_val)
gridsearch.fit(x_train_split, y_train_split, eval_set=eval_set, verbose=False)

# make predictions for test data

y_pred = model.predict(x_val)

print("--- mean_squared_error ", compute_mean_squared_error(y_pred, y_val))

y_test = model.predict(x_test)
y_test = format_output(y_test)
y_test.to_csv("submission/result_catboost.csv")


