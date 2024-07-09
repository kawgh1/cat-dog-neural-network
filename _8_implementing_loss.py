import numpy as np

# softmax_outputs = [[0.7, 0.1, 0.2],
#                    [0.1, 0.5, 0.4],
#                    [0.02, 0.9, 0.08]]

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# Classes:
# 0 -- dog
# 1 -- cat
# 2 -- human

class_targets = [0, 1, 1]

# Confidence on the correct labels:
# 0.7
# 0.5
# 0.9

# this line is saying: 
# "for the 0th row, give me the 0th element in softmax_outputs"
# "for the 1th row, give me the 1th element in softmax_outputs"
# "for the 2th row, give me the 1th element in softmax_outputs"

print(softmax_outputs[[0, 1 , 2], class_targets])

# [0.7, 0.5, 0.9]

# WARNING: The natural log of 0 is -infinity

# not handling for this can kill your model

# One way we can handle for taking the log of 0 is by clipping the values
# outputted from our softmax function (softmax_outputs)
#
# here we are CLIPPING the values in softmax_outputs:
# 
# [[0.7, 0.1, 0.2],
#  [0.1, 0.5, 0.4],
#  [0.02, 0.9, 0.08]]
# 
#  to be not less than e^-7 (0.000912)
#  and not greater than 1-e^-7 (.999088)

# y_pred_clipped = np.clip(softmax_outputs, 1e-7, 1-1e-7)
# 
# this guarantees we have no 0s in our output data before we calculate
#
# Categorical Cross Entropy Loss


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):

    '''
    y_pred = the values from the Neural Network (output from the Softmax function)
    y_true = the target values, the values we are training off of, that we know are correct
    '''
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # prevent divide by 0
        # here we are CLIPPING the values in y_pred to be not less than e^-7 (0.000912)
        # and not greater than 1-e^-7 (.999088)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # if y_true.shape is scalar values [0, 1, 1, 0...] --> 1-D array
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # if y_true.shape is a matrix or one-hot encoding
        # [[0, 0, 1]
        #  [1, 0, 0],
        #  [1, 0, 0]] --> 2-D array
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
#----------------------------------------------------------------------------------



# Handling Accuracy

# How do we tell how accurate our Softmax output predictions were?

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

# Classes:
# 0 -- dog
# 1 -- cat
# 2 -- human

class_targets = [0, 1, 1]

# Confidence on the correct labels:
# 0.7
# 0.1
# 0.9

# we can use argmax to determine, for each sample output, if the target value had the highest probability 
# in our sample, if it did, then that is a positive prediction, if it did not, that would be a failed prediction

predictions = np.argmax(softmax_outputs, axis=1)
accuracy = np.mean(predictions == class_targets)

print('accuracy: ', accuracy)
# 0.6666667


# so our softmax_ouputs got 2 out 3 correct, or 66.67%