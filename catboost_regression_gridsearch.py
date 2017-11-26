from catboost import CatBoostRegressor

from GradienBoosting import format_output
from read_data import x_train, y_train, x_test
from utility.mean_squared_error import compute_mean_squared_error
from sklearn.model_selection import GridSearchCV


search_param = {'depth': [3,4,5,6,7],
                'learning_rate': [0.02,0.05,0.1,0.15,0.3],
                'iterations': [75,100,200,300,425]}

gridsearch = GridSearchCV(estimator = CatBoostRegressor(depth=5,
                                                          learning_rate=0.1,
                                                          loss_function='RMSE',
                                                          iterations=200, eval_metric="RMSE"),
                          param_grid = search_param)




gridsearch.fit(x_train, y_train)
print(gridsearch.best_estimator_())
# make predictions for test data

y_pred = gridsearch.predict(x_train)

print("--- mean_squared_error ", compute_mean_squared_error(y_pred, y_train))

y_test = gridsearch.predict(x_test)
y_test = format_output(y_test)
y_test.to_csv("submission/result_catboost_gridsearch.csv")


best_model = gridsearch.best_estimator_()
best_model = best_model.fit(x_train,y_train)

y_pred = best_model.predict(x_train)

print("--- mean_squared_error ", compute_mean_squared_error(y_pred, y_train))

y_test = best_model.predict(x_test)
y_test = format_output(y_test)
y_test.to_csv("submission/result_catboost_gridsearch_best.csv")


