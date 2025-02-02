import numpy as np
import pandas as pd
from estimate_price import estimate_price
from plotting import print_graph
import sys
"""
This is a simple linear regression algorithm that uses gradient descent to find the best parameters for a linear model. 
The algorithm receives two numpy arrays, x and y, an accuracy parameter that defaults to 0.01, and the maximum iterations to avoid being in an infinite loop. 
The function returns the slope and the intercept of the linear model that best fits the data.

The algorithm works as follows:
1. Check if the size of x and y is the same.
2. Initialize the parameters of the linear model to zero.
3. Reshape y to be a column vector.
4. Add a column of ones to the x matrix.
5. Calculate the predicted values of y using the current parameters.
6. Iterate until the maximum number of iterations is reached or the predicted values are close enough to the actual values.
7. Calculate the gradient of the cost function.
8. Update the parameters using the gradient and the learning rate.
9. Check if the predicted values are close enough to the actual values.
"""

def linear_regression(X:np.array, y:np.array,accuracy=0.01, max_iter=10000, learning_rate=0.01):
    if X.shape[0] != y.size:
        raise ValueError("x and y must have the same size")
    m = y.size
    y = y.reshape(m, 1)
    if X.ndim == 1:
        X = X.reshape(m, 1)

    # Scale X to improve convergence
    print ((X.mean(axis = 0)/ X.std(axis = 0)).sum())
    x = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
    x = np.concat(( np.ones((m, 1)) ,x), axis = 1)
    theta = np.zeros((x.shape[1],1))
    y_predicted = np.zeros(y.shape)
    prev_los = float('inf')
    for _ in range(max_iter):
        #The following line is the natural instrucction to use in this algorithm but in therms to follow the subject, 
        # I will use a loop to calculate the y_predicted using the estimate price function.
        y_predicted = np.dot(x, theta)
        #for i in range(m):
        #    mileage = x[i,1]
        #    y_predicted [i] = estimate_price(mileage, theta[0,0], theta[1,0])
        gradient = np.dot(x.T, (y_predicted - y)) / m
        theta = theta - gradient * learning_rate
        loss = np.pow(y_predicted - y, 2).sum() / m
        if abs(loss- prev_los) < accuracy:
            break
        prev_los = loss
    print(f"Loss: {loss}")
    print(f"theta: {theta}")
    print(f"factores1: {theta[1:,0] / np.std(X, axis = 0)}")
    print(f"factores0: {theta[0,0] - np.mean(X, axis = 0) / np.std(X, axis = 0)}")
    theta = np.concat([theta[0, ] - np.dot(theta[1:,0], np.mean(X, axis = 0) / np.std(X, axis = 0)), theta[1:,0] / np.std(X, axis = 0)])
    print(f"theta: {theta }")
    #print_graph(x[:,1], y, y_predicted)
        #input("Press Enter to continue...")
    return theta
if len(sys.argv) != 2:
    print("You must provide a csv file as an argument")
    sys.exit()
else:
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        print("The file provided does not exist or is not a csv file")
        sys.exit()
    x = np.array([data['km'], np.power(data['km'], 2)]).T
    y = np.array(data['price'])
    theta = linear_regression(x, y)
    print(f"theta0: {theta[0]} theta1: {theta[1]}")
    m = y.size
    if x.ndim == 1:
        x = x.reshape(m, 1)
    x = np.concat(( np.ones(m).reshape(m,1) ,x), axis = 1)
    y_predicted = np.dot(x, theta).reshape(m, 1)
    y = y.reshape(m, 1)
    print_graph(x[:,1:], y, y_predicted)