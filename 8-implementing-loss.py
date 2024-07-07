import numpy as np

# softmax_outputs = [[0.7, 0.1, 0.2],
#                    [0.1, 0.5, 0.4],
#                    [0.02, 0.9, 0.08]]

softmax_outputs = np.array[[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

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
        # here we are CLIPPING the values in y_pred to be not less than e^-7 (0.000912)
        # and not greater than 1-e^-7 (.999088)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
