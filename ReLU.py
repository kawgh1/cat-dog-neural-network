##########################
#
# Layers as Objects
#
###########################

import numpy as np
np.random.seed(0)

# inputs -- "hidden layer"
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]] # 3x4 Matrix

class Layer_Dense:

    # initialize a layer of matrix size n_inputs x n_neurons
    # using andom numbers between -1 and 1 under a Gaussian distribution
    #
    # n_inputs = the number of features per input --> 4 in the case above ^ [1, 2, 3, 2.5]
    #
    # n_neurons can be any number we want, how deep should the layer be?
    #
    #     ex.  print(0.10 * np.random.randn(4,3))
    #
    #    [[ 0.17640523  0.04001572  0.0978738 ]
    #     [ 0.22408932  0.1867558  -0.09772779]
    #     [ 0.09500884 -0.01513572 -0.01032189]
    #     [ 0.04105985  0.01440436  0.14542735]]
    #
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward propogation of inputs through a layer
    #
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



class Acivation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        

# demo
#
# making our neural layers into objects "Layer_Dense" makes them more dynamic
# and easier to work with

layer1 = Layer_Dense(4,5)
# the only requirement for Layer 2 is that the output shape from Layer 1 
# must match the n_inputs for Layer 2 --> '5' in this case

layer2 = Layer_Dense(5,2)

layer1.forward(X)
print("\n Layer 1 output: \n\n", layer1.output)

layer2.forward(layer1.output)
print("\n Layer 2 output: \n\n", layer2.output)

#  Layer 1 output: 

#  [[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
#  [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
#  [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]

#  Layer 2 output: 

#  [[ 0.148296   -0.08397602]
#  [ 0.14100315 -0.01340469]
#  [ 0.20124979 -0.07290616]]



### Rectified Linear Activation Function (ReLU) Example

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]] # 3x4 Matrix

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

### Rectified Linear Activation Function (ReLU)
'''
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

'''
# simplified
for i in inputs:
    output.append(max(0,i))


print(output) # [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]