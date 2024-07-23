



import numpy as np
from _5_create_spiral_data import create_data
from _12_full_code_to_now import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy


# # SGD optimizer
# class Optimizer_SGD:

#     # Initialize optimizer - set settings,
#     # learning rate of 1. is default for this optimizer
#     def __init__(self, learning_rate=1., decay=0.):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0

#     # Call once before any parameter updates
#     def pre_update_params(self):
#         if self.decay:
#             self.current_learning_rate = (self.learning_rate * 
#                 (1 / (1 + self.decay * self.iterations)))

#     # Update parameters
#     def update_params(self, layer):
#         layer.weights += -self.current_learning_rate * layer.dweights
#         layer.biases += -self.current_learning_rate * layer.dbiases

#     # Call once after any parameter updates
#     def post_update_params(self):
#         self.iterations += 1

'''

'''

# # Create dataset
# X, y = create_data(samples=100, classes=3)

# # Create Dense layer with 2 input features and 64 output values
# dense1 = Layer_Dense(2, 64)

# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()

# # Create second Dense layer with 64 input features (as we take output
# # of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(64, 3)

# # Create Softmax classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# # Create optimizer --> .01
# optimizer = Optimizer_SGD(decay=1e-3)

# # Train in loop
# for epoch in range(10001):

#     # Perform a forward pass of our training data through this layer
#     dense1.forward(X)

#     # Perform a forward pass through activation function
#     # takes the output of first dense layer here
#     activation1.forward(dense1.output)

#     # Perform a forward pass through second Dense layer
#     # takes outputs of activation function of first layer as inputs
#     dense2.forward(activation1.output)

#     # Perform a forward pass through the activation/loss function
#     # takes the output of second dense layer here and returns loss
#     loss = loss_activation.forward(dense2.output, y)

#     # Calculate accuracy from output of activation2 and targets
#     # calculate values along first axis
#     predictions = np.argmax(loss_activation.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions==y)

#     if not epoch % 500:
#         print(f'epoch: {epoch}, ' +
#               f'accuracy: {accuracy:.3f}, ' +
#               f'loss: {loss:.3f}, ' +
#               f'learning rate: {optimizer.current_learning_rate}')
#     # Backward pass
#     loss_activation.backward(loss_activation.output, y)
#     dense2.backward(loss_activation.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)

#     # Update weights and biases
#     optimizer.pre_update_params()
#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)
#     optimizer.post_update_params()


'''
epoch: 0, accuracy: 0.347, loss: 1.099, learning rate: 1.0
epoch: 500, accuracy: 0.413, loss: 1.065, learning rate: 0.1669449081803005
epoch: 1000, accuracy: 0.413, loss: 1.062, learning rate: 0.09099181073703366
epoch: 1500, accuracy: 0.423, loss: 1.061, learning rate: 0.06253908692933083
epoch: 2000, accuracy: 0.417, loss: 1.061, learning rate: 0.047641734159123386
epoch: 2500, accuracy: 0.420, loss: 1.060, learning rate: 0.03847633705271258
epoch: 3000, accuracy: 0.417, loss: 1.060, learning rate: 0.03226847370119393
epoch: 3500, accuracy: 0.420, loss: 1.059, learning rate: 0.027785495971103084
epoch: 4000, accuracy: 0.423, loss: 1.059, learning rate: 0.02439619419370578
epoch: 4500, accuracy: 0.420, loss: 1.059, learning rate: 0.021743857360295715
epoch: 5000, accuracy: 0.420, loss: 1.059, learning rate: 0.019611688566385566
epoch: 5500, accuracy: 0.420, loss: 1.059, learning rate: 0.01786033220217896
epoch: 6000, accuracy: 0.417, loss: 1.058, learning rate: 0.016396130513198885
epoch: 6500, accuracy: 0.423, loss: 1.058, learning rate: 0.015153811183512654
epoch: 7000, accuracy: 0.423, loss: 1.058, learning rate: 0.014086491055078181
epoch: 7500, accuracy: 0.423, loss: 1.058, learning rate: 0.013159626266614028
epoch: 8000, accuracy: 0.423, loss: 1.058, learning rate: 0.012347203358439314
epoch: 8500, accuracy: 0.420, loss: 1.058, learning rate: 0.01162925921618793
epoch: 9000, accuracy: 0.420, loss: 1.058, learning rate: 0.010990218705352238
epoch: 9500, accuracy: 0.420, loss: 1.057, learning rate: 0.010417751849150954
epoch: 10000, accuracy: 0.420, loss: 1.057, learning rate: 0.009901970492127933
'''

'''

This model definitely got stuck, and the reason is almost certainly 
because the learning rate decayed far too quickly and became too small, 
trapping the model in some local minimum. 

This is most likely why, rather than wiggling, our accuracy and loss stopped changing at all.
We can, instead, try to decay a bit slower by making our decay a smaller number. 
For example, let’s go with 1e-3 (0.001):

'''

'''

epoch: 0, accuracy: 0.347, loss: 1.099, learning rate: 1.0
epoch: 500, accuracy: 0.420, loss: 1.058, learning rate: 0.66711140760507
epoch: 1000, accuracy: 0.437, loss: 1.044, learning rate: 0.5002501250625312
epoch: 1500, accuracy: 0.500, loss: 1.011, learning rate: 0.4001600640256102
epoch: 2000, accuracy: 0.520, loss: 0.985, learning rate: 0.33344448149383127
epoch: 2500, accuracy: 0.543, loss: 0.959, learning rate: 0.2857959416976279
epoch: 3000, accuracy: 0.573, loss: 0.938, learning rate: 0.25006251562890724
epoch: 3500, accuracy: 0.573, loss: 0.923, learning rate: 0.22227161591464767
epoch: 4000, accuracy: 0.557, loss: 0.911, learning rate: 0.2000400080016003
epoch: 4500, accuracy: 0.570, loss: 0.890, learning rate: 0.18185124568103292
epoch: 5000, accuracy: 0.593, loss: 0.862, learning rate: 0.16669444907484582
epoch: 5500, accuracy: 0.607, loss: 0.836, learning rate: 0.15386982612709646
epoch: 6000, accuracy: 0.633, loss: 0.810, learning rate: 0.1428775539362766
epoch: 6500, accuracy: 0.653, loss: 0.785, learning rate: 0.13335111348179757
epoch: 7000, accuracy: 0.673, loss: 0.756, learning rate: 0.12501562695336915
epoch: 7500, accuracy: 0.680, loss: 0.735, learning rate: 0.11766090128250381
epoch: 8000, accuracy: 0.697, loss: 0.715, learning rate: 0.11112345816201799
epoch: 8500, accuracy: 0.690, loss: 0.694, learning rate: 0.10527423939362038
epoch: 9000, accuracy: 0.713, loss: 0.676, learning rate: 0.1000100010001
epoch: 9500, accuracy: 0.707, loss: 0.662, learning rate: 0.09524716639679968
epoch: 10000, accuracy: 0.713, loss: 0.649, learning rate: 0.09091735612328393

'''


'''

In this case, we’ve achieved our lowest loss and highest accuracy thus far, 
but it still should be possible to find parameters that will give us even better results. 
For example, you may suspect that the initial learning rate is too high. 
It can make for a great exercise to attempt to find better settings. 
Feel free to try!

Stochastic Gradient Descent with learning rate decay can do fairly well 
but is still a fairly basic optimization method that only follows a gradient 
without any additional logic that could potentially help the model 
find the global minimum to the loss function. One option for improving 
the SGD optimizer is to introduce momentum.

Momentum creates a rolling average of gradients over some number of updates 
and uses this average with the unique gradient at each step. 
Another way of understanding this is to imagine a ball going down a hill — 
even if it finds a small hole or hill, momentum will let it go straight through it 
towards a lower minimum — the bottom of this hill. 

This can help in cases where you’re stuck in some local minimum (a hole), 
bouncing back and forth. With momentum, a model is more likely to pass through local minimums, 
further decreasing loss. 

Simply put, momentum may still point towards the global gradient descent direction.

'''

###################################################################################
'''

We utilize momentum by setting a parameter between 0 and 1, 
representing the fraction of the previous parameter update to retain, 
and subtracting (adding the negative) our actual gradient, 
multiplied by the learning rate (like before), from it. 

The update contains a portion of the gradient from preceding steps as our momentum 
(direction of previous changes) and only a portion of the current gradient; 
together, these portions form the actual change to our parameters 
and the bigger the role that momentum takes in the update, 
the slower the update can change the direction. 

When we set the momentum fraction too high, the model might stop learning at all 
since the direction of the updates won’t be able to follow the global gradient descent. 
The code for this is as follows:

'''

# weight_updates = (self.momentum * layer.weight_momentums - 
#                  self.current_learning_rate * layer.dweights)

'''
The hyperparameter, self.momentum, is chosen at the start and the layer.weight_momentums 
start as all zeros but are altered during training as:
'''

# layer.weight_momentums = weight_updates

'''
This means that the momentum is always the previous update to the parameters. 
We will perform the same operations as the above with the biases. 
We can then update our SGD optimizer class’ update_params method 
with the momentum calculation, applying with the parameters, 
and retaining them for the next steps as an alternative chain of operations 
to the current code. 

The difference is that we only calculate the updates and we add these updates 
with the common code:
'''

# Update parameters
def update_params(self, layer):

    # If we use momentum
    if self.momentum:

        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # The array doesn't exist for biases yet either.
            layer.bias_momentums = np.zeros_like(layer.biases)

        # Build weight updates with momentum - take previous
        # updates multiplied by retain factor and update with
        # current gradients
        weight_updates = \
            self.momentum * layer.weight_momentums - \
            self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates

        # Build bias updates
        bias_updates = \
            self.momentum * layer.bias_momentums - \
            self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates

    # Vanilla SGD updates (as before momentum update)
    else:
        weight_updates = -self.current_learning_rate * \
                            layer.dweights
        bias_updates = -self.current_learning_rate * \
                        layer.dbiases

    # Update weights and biases using either
    # vanilla or momentum updates
    layer.weights += weight_updates
    layer.biases += bias_updates


'''
Making our full SGD optimizer class:
'''

# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

#######################################################################
# Same test run using Stochastic Gradient Descent (SGD) 
# ------------------------ with a decaying learning rate
# ------------------------ and momentum
########################################################################

'''
Let’s show an example illustrating how adding momentum changes the learning process. 
Keeping the same starting learning rate (1) and decay (1e-3) 
from the previous training attempt and using a momentum of 0.5:
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
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.5)

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
              f'loss: {loss:.3f}, ' +
              f'learning rate: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

'''
epoch: 0, accuracy: 0.347, loss: 1.099, learning rate: 1.0
epoch: 500, accuracy: 0.463, loss: 1.029, learning rate: 0.66711140760507
epoch: 1000, accuracy: 0.487, loss: 0.958, learning rate: 0.5002501250625312
epoch: 1500, accuracy: 0.513, loss: 0.917, learning rate: 0.4001600640256102
epoch: 2000, accuracy: 0.550, loss: 0.882, learning rate: 0.33344448149383127
epoch: 2500, accuracy: 0.573, loss: 0.851, learning rate: 0.2857959416976279
epoch: 3000, accuracy: 0.603, loss: 0.803, learning rate: 0.25006251562890724
epoch: 3500, accuracy: 0.617, loss: 0.787, learning rate: 0.22227161591464767
epoch: 4000, accuracy: 0.677, loss: 0.711, learning rate: 0.2000400080016003
epoch: 4500, accuracy: 0.710, loss: 0.681, learning rate: 0.18185124568103292
epoch: 5000, accuracy: 0.713, loss: 0.652, learning rate: 0.16669444907484582
epoch: 5500, accuracy: 0.713, loss: 0.612, learning rate: 0.15386982612709646
epoch: 6000, accuracy: 0.747, loss: 0.573, learning rate: 0.1428775539362766
epoch: 6500, accuracy: 0.777, loss: 0.557, learning rate: 0.13335111348179757
epoch: 7000, accuracy: 0.757, loss: 0.527, learning rate: 0.12501562695336915
epoch: 7500, accuracy: 0.800, loss: 0.498, learning rate: 0.11766090128250381
epoch: 8000, accuracy: 0.820, loss: 0.477, learning rate: 0.11112345816201799
epoch: 8500, accuracy: 0.833, loss: 0.459, learning rate: 0.10527423939362038
epoch: 9000, accuracy: 0.843, loss: 0.442, learning rate: 0.1000100010001
epoch: 9500, accuracy: 0.850, loss: 0.426, learning rate: 0.09524716639679968
epoch: 10000, accuracy: 0.863, loss: 0.414, learning rate: 0.09091735612328393
'''

'''
The model achieved the lowest loss and highest accuracy that we’ve seen so far, 
but can we do even better? Sure we can! 

Let’s try to set the momentum to 0.9:
'''

