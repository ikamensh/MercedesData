import matplotlib.pyplot as plt
import numpy as np
import _tkinter
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs


'''
Utility class that helps visualize the input data. Since input is multidimensional, it simply does PCA to reduce it to 2d.
Then it plots it. The axis have little meaning but the colors do (see more here: http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/).
Proof  data is clustered, as the data point with lowest 10% values are of the same color, same for points with y values between 10% and 20% etc.
'''
def visualize_data(data_points, data_class_values):
    data_class_values = normalize(data_class_values)
    data_class_values = 10 * data_class_values
    data_class_values = np.int_(data_class_values)

    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(data_points))

    plt.scatter(transformed[data_class_values == 9][0], transformed[data_class_values == 9][1], label='90', c='#9af9d8')
    plt.scatter(transformed[data_class_values == 8][0], transformed[data_class_values == 8][1], label='80', c='blue')
    plt.scatter(transformed[data_class_values == 7][0], transformed[data_class_values == 7][1], label='70', c='lightgreen')
    plt.scatter(transformed[data_class_values == 6][0], transformed[data_class_values == 6][1], label='60', c='green')
    plt.scatter(transformed[data_class_values == 5][0], transformed[data_class_values == 5][1], label='50', c='cyan')
    plt.scatter(transformed[data_class_values == 4][0], transformed[data_class_values == 4][1], label='40', c='magenta')
    plt.scatter(transformed[data_class_values == 3][0], transformed[data_class_values == 3][1], label='30', c='yellow')
    plt.scatter(transformed[data_class_values == 2][0], transformed[data_class_values == 2][1], label='20', c='black')
    plt.scatter(transformed[data_class_values == 1][0], transformed[data_class_values == 1][1], label='10', c='red')
    plt.scatter(transformed[data_class_values == 0][0], transformed[data_class_values == 0][1], label='0', c='#d1d2d6')

    plt.legend()
    plt.show()


def normalize(data):
    return (data - data.min())/(data.max() - data.min())
