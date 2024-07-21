import numpy as np

'''
Why are the weights transposed?

You might already see where we are going with this — 

*** the sum of the multiplication of the elements is the dot product. ***

We can achieve the same result by using the np.dot method. 
For this to be possible, we need to match the “inner” shapes 
and decide the first dimension of the result, 
which is the first dimension of the first parameter. 

We want the output of this calculation to be of the shape of the gradient 
from the subsequent function — recall that we have one partial derivative 
for each neuron and multiply it by the neuron’s partial derivative 
with respect to its input. 

We then want to multiply each of these gradients with each of the partial derivatives 
that are related to this neuron’s inputs, and we already noticed that they are rows. 

The dot product takes rows from the first argument and columns from the second 
to perform multiplication and sum; thus, we need to transpose the weights.
'''


# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation

# Backward pass - for this example we're using ReLU's output
# as passed-in gradients (we're minimizing this output)
dvalues = relu_outputs

# Backpropagation and optimization

# ReLU activation's derivative with the chain rule applied
drelu = dvalues.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)


# >>>
# [[ 0.179515   0.5003665 -0.262746 ]
#  [ 0.742093  -0.9152577 -0.2758402]
#  [-0.510153   0.2529017  0.1629592]
#  [ 0.971328  -0.5021842  0.8636583]]
# [[1.98489  2.997739 0.497389]]

'''
In this code, we replaced the plain Python functions with NumPy variants, 
created example data, calculated the forward and backward passes, 
and updated the parameters. Now we will update the dense layer 
and ReLU() activation code with a backward method (for backpropagation), 
which we’ll call during the backpropagation phase of our model.
'''
# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''
During the forward method for our Layer_Dense class, 
we will want to remember what the inputs were 
(recall that we’ll need them when calculating the partial derivative 
with respect to weights during backpropagation), which can be easily 
implemented using an object property (self.inputs):
'''
# Dense layer
class Layer_Dense:
    ...
    # Forward pass
    def forward(self, inputs):
        ...
        self.inputs = inputs

'''
Next, we will add our backward pass (backpropagation) code that 
we developed previously into a new method in the layer class, 
which we’ll call backward:
'''
class Layer_Dense:
    ...
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

'''
We then do the same for our ReLU class:
'''
# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

'''
By this point, we’ve covered everything we need to perform backpropagation, 
except for the derivative of the Softmax activation function 
and the derivative of the cross-entropy loss function.
'''
