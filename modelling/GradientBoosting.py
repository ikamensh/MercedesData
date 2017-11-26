import pandas as pd
from sklearn import ensemble
from preprocessing.x_train_preparer import preprocess_x, extract_y
from sklearn.model_selection import GridSearchCV

x_train = preprocess_x("../data/train.csv")
x_test = preprocess_x("../data/test.csv")
sample_sub = pd.read_csv("../data/sample_submission.csv")

x_train, y_train = extract_y(x_train)

#Model training
model = ensemble.GradientBoostingRegressor()
model.fit(x_train,y_train)

#GridSearch
search_param = {'n_estimators': [100,200,500],
                'max_depth': [2,5,20],
                'min_samples_split': [2,10,50],
                'learning_rate': [0.01,0.1]}

gridsearch = GridSearchCV(estimator = ensemble.GradientBoostingRegressor(n_estimators = 2,
                                                                         max_depth = 2,
                                                                         min_samples_split = 100,
                                                                         learning_rate = 0.1,),
                          param_grid = search_param)

gridsearch.fit(x_train,y_train)
best_model = gridsearch.best_estimator_()
best_model = best_model.fit(x_train,y_train)


#Model validation
y_test = pd.DataFrame(best_model.predict(x_test))
y_test.columns= ['y']
y_test['ID'] = sample_sub.index.values
#Read ID from sample submission
y_test.set_index('ID',inplace=True)
#Save csv for submission
y_test.to_csv('../submission/submission_neu.csv')