'''
find the natural logarithm of a number b is solving for x:
    e ^ x = b

'''

import math

softmax_output = [0.7, 0.1, 0.2] # example output from a softmax function

target_output = [1, 0, 0] # one-hot encoding

loss = - (math.log(softmax_output[0]) * target_output[0] +
          math.log(softmax_output[1]) * target_output[1] +
          math.log(softmax_output[2]) * target_output[2] )

# loss = - ( -0.356674943938 + 0 + 0)
# loss = 0.35667494393873245
