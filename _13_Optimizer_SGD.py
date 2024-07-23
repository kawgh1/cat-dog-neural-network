'''
In the case of Stochastic Gradient Descent, 
we choose a learning rate, such as 1.0. 
We then subtract the learning_rate · parameter_gradients 
from the actual parameter values. 
'''
import numpy as np
from _5_create_spiral_data import create_data
from _12_full_code_to_now import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy


# SGD Optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1.0 is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# Example usage

# Create dataset
X, y = create_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU Activation
activation1 = Activation_ReLU()

# Create second dense layer with 64 input features (made of the
# 64 output features from the first layer) and 3 output values
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create Optimizer
optimizer = Optimizer_SGD()

'''
next we do our forward pass
'''

# Then perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a dorward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# print('\n\nloss : ', loss, '\n\n') 
# loss: 1.0986104

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# print('\n\naccuracy: ', accuracy, '\n\n') 
# accuracy:  0.3466666666666667 

'''
Next we do our backward pass
'''

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Then we finally use our optimizer to update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)


'''
This is everything we need to train our model! 


But why would we only perform this optimization once, 
when we can perform it lots of times by leveraging Python’s 
looping capabilities? 

We will repeatedly perform a forward pass, backward pass, 
and optimization until we reach some stopping point. 
Each full pass through all of the training data is called an epoch. 
In most deep learning tasks, a neural network will be trained 
for multiple epochs, though the ideal scenario would be to have 
a perfect model with ideal weights and biases after only one epoch. 

To add multiple epochs of training into our code, 
we will initialize our model and run a loop around all the 
code performing the forward pass, backward pass, and optimization 
calculations:

'''

# Create dataset
X, y = create_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output 
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGD(learning_rate=1.0)

# Train in loop
for epoch in range(10001):

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)


    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 500:
        print(f'epoch: {epoch}, ' +
              f'accuracy: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

'''
    epoch: 0, accuracy: 0.367, loss: 1.099
    epoch: 500, accuracy: 0.420, loss: 1.059
    epoch: 1000, accuracy: 0.470, loss: 1.023
    epoch: 1500, accuracy: 0.450, loss: 1.000
    epoch: 2000, accuracy: 0.457, loss: 0.980
    epoch: 2500, accuracy: 0.527, loss: 0.929
    epoch: 3000, accuracy: 0.553, loss: 0.885
    epoch: 3500, accuracy: 0.583, loss: 0.851
    epoch: 4000, accuracy: 0.617, loss: 0.771
    epoch: 4500, accuracy: 0.630, loss: 0.730
    epoch: 5000, accuracy: 0.713, loss: 0.650
    epoch: 5500, accuracy: 0.673, loss: 0.660
    epoch: 6000, accuracy: 0.640, loss: 0.842
    epoch: 6500, accuracy: 0.660, loss: 0.702
    epoch: 7000, accuracy: 0.747, loss: 0.568
    epoch: 7500, accuracy: 0.740, loss: 0.543
    epoch: 8000, accuracy: 0.810, loss: 0.466
    epoch: 8500, accuracy: 0.783, loss: 0.501
    epoch: 9000, accuracy: 0.753, loss: 0.532
    epoch: 9500, accuracy: 0.770, loss: 0.514
    epoch: 10000, accuracy: 0.780, loss: 0.512
'''

