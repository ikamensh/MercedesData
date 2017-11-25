import pandas as pd
from sklearn import ensemble
from preprocessing.x_train_preparer import preprocess_x, extract_y

x_train = preprocess_x("data/train.csv")
x_test = preprocess_x("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")

x_train, y_train = extract_y(x_train)

#Model training
model = ensemble.GradientBoostingRegressor()
model.fit(x_train,y_train)

#Model validation
y_test = pd.DataFrame(model.predict(x_test))
y_test.columns= ['y']
y_test['ID'] = sample_sub.index.values
#Read ID from sample submission
y_test.set_index('ID',inplace=True)
#Save csv for submission
y_test.to_csv('../submission/submission.csv')