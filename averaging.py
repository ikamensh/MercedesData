import pandas as pd

predictions1 = pd.read_csv("submission/result_catboost.csv", index_col="ID")
predictions2 = pd.read_csv("submission/result_mlp_elu_early.csv", index_col="ID")
predictions3 = pd.read_csv("submission/submission_grad_class.csv", index_col="ID")

#weitghted averaging
predictions_list = [predictions1, predictions2, predictions3]
coefs=[9, 4, 6]

weighted_avg = sum( coefs[i]*pred for i, pred in enumerate(predictions_list)) / sum(coefs)
print(weighted_avg.tail())

weighted_avg.to_csv("submission/avg.csv")