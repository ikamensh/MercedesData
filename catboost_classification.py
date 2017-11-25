from preprocessing.x_train_preparer import preprocess_x, extract_y
from read_data import x_train_split, x_val, y_train_split, y_val




import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss



model = CatBoostClassifier(max_depth=5,
                      learning_rate=0.02,
                      objective='Logloss',
                      iterations=650, eval_metric="RMSE")

print(model.get_params().keys())

eval_set = (x_val, y_val)
model.fit(x_train_split, y_train_split, eval_set=eval_set, verbose=True)

# make predictions for test data

y_pred = model.predict(x_val)
ll = log_loss(y_val, y_pred)
print("Log_loss: %f" % ll)
print(model)

y_test = model.predict(x_test)
np.savetxt("predictions/result_catboost.csv", y_test)

