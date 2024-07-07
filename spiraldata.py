from create_spiral_data import create_data

# Spiral Data fed into our basic neural net of 2 layers demo

import numpy as np
np.random.seed(0)

# inputs -- "hidden layer"

# "X" denotes your feature set
# "y" denotes the target or classification

# So here we are adding 100 features in our feature set and 3 classes

X, y = create_data(samples=100,classes=3)

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

# Softmax activation funciton
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # input ex. [4.8, -1.21, 2.385]
        # Get unnormalized input values, cleared of negative values by exponentiation using Euler's number under the hood
            # Note: we subtract the max value in the row from all the values to avoid an overflow error during exponentiation
        exponentiated_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exponentiated_values / np.sum(exponentiated_values, axis=1, keepdims=True)
        # output values will be probabilities adding up to 1 for each row,
        # that each of the inputs in that row is the "correct" value for our classification
        self.output = probabilities
        # ex.)  [0.9159006914291169, 0.0022477010612691308, 0.08185160750961398]
        # thus, the first or 0th input to this function (4.8) is 91% confidence the "correct" classification value
        # associated with our classification model
        # this is less obvious when the inputs might be [3, 3.4, 2.9] --> how do we calculate the probabilities
        # that 3.4 is the correct one, how do we quanitify our confidence level of that --> using softmax

# demo
#

# setup First Layer will just be our dataset as input
# with 2 input features (x and y coordinates) and 3 output values
layer1 = Layer_Dense(2,3)

# To be used with Layer 1
activation1 = Acivation_ReLU()

#Create a second Dense Layer with 3 input features and 3 output values
layer2 = Layer_Dense(3,3)

# Create Softmax activation to be used with Layer2
activation2 = Activation_Softmax()

# begin forward propogation

# Make a forward pass of our training data through this layer
layer1.forward(X)

print("\n\n layer 1 before ReLU activation function: \n\n", layer1.output[:5])

'''
 layer 1 before activation function: 

 [[ 0.          0.          0.        ]
 [-0.00104752  0.00113954 -0.00047984]
 [-0.00274148  0.00317291 -0.00086922]
 [-0.00421884  0.00526663 -0.00055913]
 [-0.00577077  0.00714014 -0.0008943 ]]
'''
# Forward pass through the activation function
# takes in output from previous layer
activation1.forward(layer1.output)
print("\n\n layer 1 after ReLU activation function: \n\n", activation1.output[:5])

'''
 layer 1 after activation function: 

 [[0.         0.         0.        ]
 [0.         0.00113954 0.        ]
 [0.         0.00317291 0.        ]
 [0.         0.00526663 0.        ]
 [0.         0.00714014 0.        ]]
 '''

# here we can see outputs less than 0 have been changed to 0 by our ReLU activation function
# these outputs will then go in, as adjusted, to the next hidden layer

# begin Layer 2
layer2.forward(activation1.output)

print("\n\n layer 2 before Softmax activation function: \n\n", layer2.output[:5])

'''
 layer 2 before Softmax activation function: 

 [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [-1.81839690e-05 -1.52357751e-05  1.22812795e-04]
 [-5.06312943e-05 -4.24223674e-05  3.41958940e-04]
 [-8.40413550e-05 -7.04156053e-05  5.67607309e-04]
 [-1.13937683e-04 -9.54647970e-05  7.69524259e-04]]
'''

# begin Softmax
activation2.forward(layer2.output)

print("\n\n layer 2 after Softmax activation function: \n\n", activation2.output[:5])

'''
 layer 1 after activation function: 

 [[0.33333333 0.33333333 0.33333333]
 [0.33331734 0.33331832 0.33336434]
 [0.3332888  0.33329153 0.33341967]
 [0.33325941 0.33326395 0.33347665]
 [0.33323311 0.33323926 0.33352763]]
 '''

# These values are normally distributed (all are about 1/3 or .33) because the sample data was normally distributed
# to begin with. So we would expect a normalized output from Softmax if we gave it normalized inputs.