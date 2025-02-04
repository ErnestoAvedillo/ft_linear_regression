import json
import numpy as np
from math import pow
def estimate_price(mileage):
    arguments_file = "arguments.json"
    try:
        with open(arguments_file, "r", encoding="utf-8") as my_file:
            data = json.load(my_file)
            theta = np.array(data["arguments"])
    except:
        print("No file 'arguments.json' found, arguments 0,0 asumed.")
        theta = np.zeros(2)
    mileage_array = np.ones(1)
    for i in range(1, theta.size):
        mileage_array = np.concat((mileage_array, np.array([pow(mileage,i)])), axis = 0)
    print (theta)
    print (mileage_array)
    price = np.dot(theta, mileage_array)
    print(f"The estimate price for the car is {round(price,2)}")

    return  
    
mileage = input ("Enter a mileage of your car:")
if not mileage.isdigit():
   print ("La entrada ha de ser numérica.")
   exit(1)
estimate_price(float(mileage))

    