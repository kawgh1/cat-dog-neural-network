'''
___________________________________________________
Backpropagation with ReLU in a Small Neural Network

___________________________________________________
--- Network Architecture

Input Layer: 2 neurons
Hidden Layer: 2 neurons with ReLU activation
Output Layer: 1 neuron with linear activation



input1 -\--/- hidden1 \
         \/            \ output1
         /\            /
input2 -/--\- hidden2 /

____________________________________________________
--- Explanation of Weights and Biases

Weights from Input Layer to Hidden Layer (W1):

    Shape: (2, 2)
    Explanation: There are 2 input neurons and 2 hidden neurons. 
    Each hidden neuron receives inputs from both input neurons. 
    Thus, W1 has 2 rows (one for each input neuron) and 
    2 columns (one for each hidden neuron).

Biases for Hidden Layer (b1):

    Shape: (1, 2)
    Explanation: There are 2 hidden neurons, each having 
    its own bias. Thus, b1 has 1 row and 2 columns 
    (one for each hidden neuron).

Weights from Hidden Layer to Output Layer (W2):

    Shape: (2, 1)
    Explanation: There are 2 hidden neurons and 1 output neuron. 
    Each output neuron receives inputs from both hidden neurons. 
    Thus, W2 has 2 rows (one for each hidden neuron) and 
    1 column (one for the output neuron).

Bias for Output Layer (b2):

    Shape: (1, 1)
    Explanation: There is 1 output neuron, which has its 
    own bias. Thus, b2 has 1 row and 1 column.

_____________________________________________________
Steps Involved

    1.) Forward Pass: 
                Compute the output of the network for a given input.

    2.) Compute Loss: 
                Calculate the error (loss) between the predicted output 
                    and the actual target using Mean Squared Error (MSE).

    3.) Backward Pass: 
                Compute the gradient of the loss with respect to 
                    each weight using the chain rule.

    4.) Update Weights: 
                Adjust the weights using the gradients to minimize the loss.

'''
import numpy as np

# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Input data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Target output
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)
# Weights from input layer of 2 inputs to hidden layer of 2 nodes
W1 = np.random.rand(2, 2) # 2x2 
# Biases for hidden layer of 2 nodes
b1 = np.random.rand(1, 2) # 1x2 
# Weights from hidden layer of 2 nodes to output layer of 1 node
W2 = np.random.rand(2, 1) # 2x1
# Bias for output layer of 1 node
b2 = np.random.rand(1, 1) # 1x1

print("\nW1 = \n", W1)
print("\nb1 = \n", b1)
print("\nW2 = \n", W2)
print("\nb2 = \n", b2)


# Learning rate
eta = 0.1
# Number of iterations for training
epochs = 10000
epoch_counter = 0
loss_counter = 0
previous_loss = 999



for epoch in range(epochs):
    epoch_counter += 1
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = z2  # Linear activation for output layer

    
    # Compute the loss (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)
    if(epoch_counter % 100 == 0):
        print("\n\n current loss: ", loss)
        # print("a2 == \n", a2)

    
    if(previous_loss <= loss):
        loss_counter += 1

    if(loss_counter > 9):
        print("\n\nprevious loss: ", previous_loss)
        print("\ncurrent loss: ", loss)
        print("\n\n10 tries did not result in a lower loss value\n\n")
        print("\n\ntotal iterations: ", epoch_counter)
        print("a2 == \n", a2)
        break
    previous_loss = loss

    
    # Backward pass
    d_loss_a2 = a2 - y
    d_loss_z2 = d_loss_a2  # since derivative of linear activation is 1
    
    d_loss_W2 = np.dot(a1.T, d_loss_z2)
    d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)
    
    d_loss_a1 = np.dot(d_loss_z2, W2.T)
    d_loss_z1 = d_loss_a1 * relu_derivative(z1)
    
    d_loss_W1 = np.dot(X.T, d_loss_z1)
    d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)
    
    # Update weights and biases by multiplying by the learning rate (0.1)
    W2 -= eta * d_loss_W2
    b2 -= eta * d_loss_b2
    W1 -= eta * d_loss_W1
    b1 -= eta * d_loss_b1

# Test the trained network
def predict(x):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = z2  # Linear activation for output layer
    return a2

# Testing the network on training data
print("\n\n\n results: \n")
for x in X:
    print(f"{x} -> {predict(x)}")
