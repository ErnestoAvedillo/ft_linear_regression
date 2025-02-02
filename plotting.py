from matplotlib import pyplot as plt
import numpy as np

def print_graph(x:np.array, y:np.array, y_predicted:np.array):
    plt.scatter(x, y, color='red', label='Real data')
    plt.plot(x, y_predicted, color='blue', label='Predicted data')
    plt.show()