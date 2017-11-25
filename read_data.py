from preprocessing.x_train_preparer import preprocess_x, extract_y

x_train = preprocess_x("data/train.csv")
x_test = preprocess_x("data/test.csv")

x_train, y_train = extract_y(x_train)

from sklearn.model_selection import train_test_split

x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.25)
