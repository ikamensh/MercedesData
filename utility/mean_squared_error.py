from sklearn.metrics import mean_squared_error

def compute_model_error(model, data, actual_values):
    prediction = model.predict(data)

    result = mean_squared_error(prediction, actual_values)

    return result

def compute_mean_squared_error(predicted, actual):
    return mean_squared_error(predicted, actual)

