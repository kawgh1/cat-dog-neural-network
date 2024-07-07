### This is a neuron

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2 # each neuron has only 1 bias

# First step for a neuron is add up all the inputs * weights + bias

output = (inputs[0] * weights[0] + 
          inputs[1] * weights[1] + 
          inputs[2] * weights[2] +
          inputs[3] * weights[3] + 
          bias)

print(output)

# output: [4.8, 1.21, 2.385]

###########################################

# 3 Neurons with 4 inputs each (a "Layer")

###########################################


### This is a layer of neurons

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5] # each neuron has only 1 bias

# First step for a neuron is add up all the inputs * weights + bias

# So you can't really change the inputs if you wanted to change the output
# you would have to adjust either the weights or the bias or both --> tuning

layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs) # output: [4.8, 1.21, 2.385]


########################################

# From   Matrices and Dot Products --> Batches

########################################
import numpy as np
print(np.__version__)

inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2

output = np.dot(weights, inputs) + bias
print(">>>")
print("Dot Product of a single layer of a single neuron with 4 inputs")
print(">>>")
print(output) # 4.8

######################################

# Dot Product of a *Layer* of neurons

######################################

inputs = [1, 2, 3, 2.5] # 1x4 Matrix

weights = [[0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]  # 3x4 Matrix

biases = [2, 3, 0.5]

# inputs is transposed into a 4x1 matrix to do the dot product
#                3x4   ·   4x1
output = np.dot(weights, inputs) + biases
# np.dot(weights, inputs) = 
#                          [ np.dot(weights[0], inputs),    # (0.2 * 1) + (0.8 * 2) + (-0.5 * 3) + (1.0 * 2.5) = 2.8 
#                            np.dot(weights[1], inputs),    # etc.
#                            np.dot(weights[2], inputs) ]   # etc.
print(">>>")
print("Dot Product of a single layer of 3 neurons with 4 inputs")
print(">>>")
print(output) # output: [4.8, 1.21, 2.385]



####################################################

# 1 Batch of inputs with a single layer of 3 neurons

#####################################################

# 1 Batch of 3 inputs

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]] # 3x4 Matrix

# 1 Layer of 3 Neurons

weights = [[0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]  # 3x4 Matrix

biases = [2, 3, 0.5]

# weights is transposed into a 4x3 matrix to do the dot product
#                3x4   ·   4x3
output = np.dot(inputs, np.array(weights).T) + biases

# Transpose weights
# [ 0.2 0.5 -0.26 ]
# [ 0.8 -0.91 -0.27 ]
# [ -0.5 0.26 0.17 ]
# [ 1.0 -0.5 0.87 ]

# Dot Product on the Transpose of weights
#
#        inputs           ·          weights T                                      +    biases       =        result 3x4 matrix
#
# [  1.0 2.0  3.0  2.5 ]      [  0.2  0.5  -0.26 ]       [  2.8  -1.79   1.885 ]                            [ 4.8    1.21   2.385 ]
# [  2.0 5.0 -1.0  2.0 ]  ·   [  0.8 -0.91 -0.27 ]   =   [  6.9  -4.81  -0.3   ]    +  [ 2 3 0.5 ]    =     [ 8.9   -1.81   0.2   ]
# [ -1.5 2.7  3.3 -0.8 ]      [ -0.5  0.26  0.17 ]       [ -0.59 -1.949 -0.474 ]                            [ 1.41   1.051  0.026 ]
#                             [  1.0 -0.5   0.87 ]

print(">>>")
print("1 Batch of inputs with a single layer of 3 neurons")
print(">>>")
print(output) # output: [4.8, 1.21, 2.385]

#           result
#
#  [[ 4.8    1.21   2.385]
#   [ 8.9   -1.81   0.2  ]
#   [ 1.41   1.051  0.026]]


########################################################

# 1 Batch of inputs with 2 layers of 3 neurons each

########################################################

# 1 Batch of 3 inputs

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]] # 3x4 Matrix

# 1st Layer of 3 Neurons

weights1 = [[0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]  # 3x4 Matrix

biases1 = [2, 3, 0.5]

# 2nd Layer of 3 Neurons

weights2 = [[0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]  # 3x4 Matrix

biases2 = [-1, 2, -0.5]

#                        3 x 4   ·   4 x 3                
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

#   layer1_outputs   result
#
#  [[ 4.8    1.21   2.385]
#   [ 8.9   -1.81   0.2  ]
#   [ 1.41   1.051  0.026]]

# layer 1 output becomes the input for layer 2

#                           3 x 3     ·     3 x 3                
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(">>>")
print("1 Batch of inputs with 2 layers of 3 neurons each")
print(">>>")
print(layer2_outputs) 

#  layer2_outputs   result
#
# [[ 0.5031  -1.04185 -2.03875]
#  [ 0.2434  -2.7332  -5.7633 ]
#  [-0.99314  1.41254 -0.35655]]