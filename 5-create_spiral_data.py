# generate a data set

import numpy as np

np.random.seed(0)

# https://cs231n.github.io/neural-networks-case-study
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    
    for class_number in range(classes):
        idx = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples) # radius
        
        t = (np.linspace(class_number * 4, (class_number+1) * 4, samples) 
             + np.random.randn(samples) * 0.2 )
        
        X[idx] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[idx] = class_number

    return X,y

import matplotlib.pyplot as plt

print("here")
X, y = create_data(100,3)

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()