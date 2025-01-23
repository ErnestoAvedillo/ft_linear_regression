import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sys

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((X.dot(theta) - y)**2)


if len(sys.argv) != 2:
    print("Usage: python main.py <grade>")
    sys.exit(1)

grade = int(sys.argv[1])

if grade not in [1, 2, 3, 4, 5]:
    print("Grade must be between 1 and 5")
    sys.exit(1)

fd = pd.read_csv('data.csv')

print(fd.head())

data = fd.to_numpy()

polynomy = np.random.rand(grade)
print (polynomy)    

