from create_spiral_data import create_data

# Spiral Data fed into our basic neural net of 2 layers demo

import numpy as np
np.random.seed(0)

# inputs -- "hidden layer"

# "X" denotes your feature set
# "y" denotes the target or classification

# So here we are adding 100 features in our feature set and 3 classes

X, y = create_data(100,3)

class Layer_Dense:

    # initialize a layer of matrix size n_inputs x n_neurons
    # using andom numbers between -1 and 1 under a Gaussian distribution
    #
    # n_inputs = the number of features per input --> 2 in the case above ^ 
    #                        because it is set of x,y coordinates on a graph
    #
    # n_neurons can be any number we want, how deep should the layer be?
    #
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward propogation of inputs through a layer
    #
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



# ReLU Acitivation Function
class Acivation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# demo
#

# setup First Layer will just be our dataset as input
#
layer1 = Layer_Dense(2,5)

activation1 = Acivation_ReLU()

# run
layer1.forward(X)

print("\n\n layer 1 before activation function: \n\n", layer1.output)
activation1.forward(layer1.output)
print("\n\n layer 1 after activation function: \n\n", activation1.output)