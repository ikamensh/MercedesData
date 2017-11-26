import pandas as pd
from sklearn import ensemble
from preprocessing.x_train_preparer import preprocess_x, extract_y

x_train = preprocess_x("data/train.csv")
x_test = preprocess_x("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv",index_col="ID")

x_train, y_train = extract_y(x_train)

#Model training
model = ensemble.GradientBoostingRegressor()
model.fit(x_train,y_train)

#Model validation
def format_output(prediction):
    prediction = pd.DataFrame(prediction)
    prediction.columns= ['y']
    prediction['ID'] = sample_sub.index.values
    #Read ID from sample submission
    prediction.set_index('ID',inplace=True)
    #Save csv for submission
    return prediction

y_test = format_output(model.predict(x_test))


y_test.to_csv('submission/submission_grad_class.csv')