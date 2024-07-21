'''
Backpropagation is a method used in neural networks to adjust the weights 
of the network to minimize the error in predictions. 

Step-by-Step Explanation

1.) Forward Pass: Compute the output of the network for a given input.
2.) Compute Loss: Calculate the error (loss) between the predicted output and the actual target.
3.) Backward Pass: Compute the gradient of the loss with respect to each weight using the chain rule.
4.)Update Weights: Adjust the weights using the gradients to minimize the loss.
Example

We'll create a simple neural network with one hidden layer and use the Mean Squared Error (MSE) as our loss function.
'''

'''
Network Architecture
Input layer: 2 neurons
Hidden layer: 2 neurons
Output layer: 1 neuron
'''

'''
Forward Pass Equations
Hidden layer activations: 𝑧1 = 𝑊1 ⋅ 𝑥 + 𝑏1
 
Hidden layer outputs:  𝑎1 = 𝜎(𝑧1) (using the sigmoid activation function)
Output layer activations: 𝑧2 = 𝑊2 ⋅ 𝑎1 + 𝑏2 
Output layer outputs: 𝑎2 = 𝜎(𝑧2)
'''

'''
Loss

Mean Squared Error (MSE):
Loss = 1/2 * ∑(𝑦 − 𝑎2) ^ 2


Backward Pass Equations
Output layer error: 𝛿2 = (𝑎2 − 𝑦) ⋅ 𝜎′(𝑧2)
Hidden layer error: 𝛿1 = (𝛿2 ⋅𝑊2) ⋅ 𝜎′(𝑧1)
'''

import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Target output
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(2, 2)
b1 = np.random.rand(1, 2)
W2 = np.random.rand(2, 1)
b2 = np.random.rand(1, 1)

# Learning rate
eta = 0.1
# Number of iterations for training
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Compute the loss (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)
    
    # Backward pass
    d_loss_a2 = a2 - y
    d_a2_z2 = sigmoid_derivative(a2)
    d_loss_z2 = d_loss_a2 * d_a2_z2
    
    d_loss_W2 = np.dot(a1.T, d_loss_z2)
    d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)
    
    d_loss_a1 = np.dot(d_loss_z2, W2.T)
    d_a1_z1 = sigmoid_derivative(a1)
    d_loss_z1 = d_loss_a1 * d_a1_z1
    
    d_loss_W1 = np.dot(X.T, d_loss_z1)
    d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= eta * d_loss_W2
    b2 -= eta * d_loss_b2
    W1 -= eta * d_loss_W1
    b1 -= eta * d_loss_b1

# Test the trained network
def predict(x):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2

# Testing the network on training data
for x in X:
    print(f"{x} -> {predict(x)}")

'''
In this example:

We define the sigmoid activation function and its derivative.
We initialize the input data X and target output y.
We initialize weights and biases with random values.
We perform the forward and backward passes iteratively for a specified number of epochs.
We update the weights and biases based on the computed gradients.
Finally, we test the trained network on the input data.
This simple example demonstrates how backpropagation works in a small neural network.

'''
