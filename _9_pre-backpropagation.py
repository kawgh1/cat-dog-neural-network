

# This is a forward pass of a single neuron using the ReLU function

# Forward pass
x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)
# -3.0 2.0 6.0 1.0

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print(z)
# 6.0

# ReLU activation function
y = max(z, 0)
print(y)
# 6.0

'''
x[0] -- 1.0   \
               -3.0  \
w[0] -- -3.0  /        \
                         \
x[1] -- -2.0  \            \
                2.0 -------- 6.0 -- ReLU = max(6,0)  =  6.0
w[1] -- -1.0  /            /  |
                         /   /
x[2] -- 3.0   \        /    /
                6.0  /    /
w[2] -- 2.0   /         /
                     /
b --------------1.0 

'''

'''
This is the full forward pass through a single neuron and a
 ReLU activation function. Let’s treat all of these chained functions 
 as one big function which takes input values (x), weights (w), and bias (b), 
 as inputs, and outputs y. 
 
 This big function consists of multiple simpler functions — 
 there is a multiplication of input values and weights, 
 sum of these values and bias, as well as a max function 
 as the ReLU activation — 3 chained functions in total.
'''

'''
This will look something like:

ReLU( Σ [inputs * weights] + bias )

or

ReLU( (x0 * x1) + (x1 * w1) + (x2 * w2) + b )

'''

'''
The first step is to backpropagate our gradients by calculating derivatives
and partial derivatives with respect to each of our parameters and inputs. 

To do this, we will use the chain rule.
'''