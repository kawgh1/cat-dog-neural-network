### This is a neuron

inputs = [1,2,3, 2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2 # each neuron has only 1 bias

# First step for a neuron is add up all the inputs * weights + bias

output = (inputs[0] * weights[0] + 
          inputs[1] * weights[1] + 
          inputs[2] * weights[2] +
          inputs[3] * weights[3] + 
          bias)

print(output)

###########################

# 3 Neurons with 4 inputs

###########################


### This is a neuron

inputs = [1,2,3, 2.5]

weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2 # each neuron has only 1 bias
bias2 = 3
bias3 = 0.5 

# First step for a neuron is add up all the inputs * weights + bias

# So you can't really change the inputs if you wanted to change the output
# you would have to adjust either the weights or the bias or both --> tuning

output = [
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1, # First neuron with 4 inputs
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2, # Second neuron with 4 inputs
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3  # Third neuron with 4 inputs
    ]

print(output)

# output [4.8, 1.21, 2.385]

