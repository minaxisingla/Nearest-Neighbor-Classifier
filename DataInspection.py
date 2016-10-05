
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

#Load training data
train_data = np.loadtxt('train.txt', delimiter=',')
#Load test data
test_data = np.loadtxt('test.txt', delimiter=',')

# Generate scatter plots for all the features with the target variable
for x in range(1,train_data.shape[1]):
    plt.figure(x)
    plt.scatter(train_data[:,0], train_data[:,x])
    
plt.show()

