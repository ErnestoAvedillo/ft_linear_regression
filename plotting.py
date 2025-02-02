from matplotlib import pyplot as plt
import numpy as np

def print_graph(x:np.array, y:np.array, y_predicted:np.array):
    print(x.shape, y.shape, y_predicted.shape)
    if x.shape[1] > 1:
        x = x[:,0].reshape(x.shape[0], 1)
    Table = np.concatenate((x, y, y_predicted), axis=1)
    Sorted_table = Table[Table[:,0].argsort()]
    print(Sorted_table.shape)
    plt.scatter(Sorted_table[:,0], Sorted_table[:,1], color='red', label='Real data')
    plt.plot(Sorted_table[:,0], Sorted_table[:,2], color='blue', label='Predicted data')
    plt.show()