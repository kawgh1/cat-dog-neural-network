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
