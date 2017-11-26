import pandas as pd


predictions1 = pd.read_csv("submission/result_mlp_elu_early.csv", index_col="ID")

predictions2 = pd.read_csv("submission/submission_grad_class.csv", index_col="ID")
predictions3 = pd.read_csv("submission/submission_neu2.csv", index_col="ID")
predictions4 = pd.read_csv("submission/submission_neu.csv", index_col="ID")

predictions5 = pd.read_csv("submission/result_catboost.csv", index_col="ID")
predictions6 = pd.read_csv("submission/result_xgboost.csv", index_col="ID")

predictions7 = pd.read_csv("submission/result_catboost_gridsearch.csv", index_col="ID")
predictions8 = pd.read_csv("submission/result_xgboost_gridsearch.csv", index_col="ID")

#weitghted averaging
predictions_list = [predictions1, predictions2, predictions3, predictions4, predictions5, predictions6, predictions7, predictions8]
coefs=[1, 3, 10, 5, 7, 4, 6, 6]

weighted_avg = sum( coefs[i]*pred for i, pred in enumerate(predictions_list)) / sum(coefs)
print(weighted_avg.tail())

weighted_avg.to_csv("submission/avg.csv")