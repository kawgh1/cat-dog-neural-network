import numpy as np
import math

# # Softmax activation funciton
# class Activation_Softmax:

#     # Forward pass
#     def forward(self, inputs):
#         # Get unnormalized probabilities
#         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         # Normalize them for each sample
#         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

#         self.output = probabilities

# Raw implementation

layer_outputs = [4.8, -1.21, 2.385] # how can we scale the "correctness" of these values for a classification model?
print("\n Unormalized values, perhaps negative numbers: \n\n", layer_outputs)

# we have 3 values in our output, obviously the highest one is the most correct, but by how much? 
# how do we measure? how do we deal with negative numbers in our output?
# 
# use exponentiation to clear negative values while keep all our numbers "meaning" or "distance" relative to each other

# E = 2.71828182846
E = math.e 

exponentiated_values = []

for output in layer_outputs:
    exponentiated_values.append(E**output)

print("\n Unormalized, Exponentiated values, cleared of negative numbers: \n\n", exponentiated_values)
#  [121.51041751873483, 0.2981972794298874, 10.859062664920513]

# Great, no negative values and we haven't lost the scale or meaning of the distance between our numbers

# next we need to create a probability output, an estimate of the probability that each number is correct, 
# with all the outputs adding up to 1 or 100%

# we can do this by, for each output row, add up the numbers in each row for a total (sum) and divide
# each number in the output array by the sum, this gives us a properly scalled probability for each of
# the output values, adding up to 1

normalized_base = sum(exponentiated_values)
normalized_values = []

for value in exponentiated_values:
    normalized_values.append(value/normalized_base)

print("\n Normalized values, cleared of negative numbers, each row adding up to 1: \n\n", normalized_values)
#  [0.9159006914291169, 0.0022477010612691308, 0.08185160750961398]

# this tells us of our original layer_outputs = [4.8, -1.21, 2.385], the first or 0th number
# is 91% chance the correct value, we have a 91% confidence score. So for that sample,
# our classification model would classify the sample output as the first or 0th classification.

# could be colors, could be animals from a picture, or whatever


