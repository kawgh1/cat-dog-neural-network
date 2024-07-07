# cat-dog-neural-network

Build a Neural Network from scratch to predict images of cats and dogs

### Based off project from [Sentdex](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)

### The book [nnfs.io](https://nnfs.io/)

### Github https://github.com/Sentdex/NNfSiX

###

- A Tensor is any object that can be represented as an array or array of arrays, etc.

### Why Batches of inputs?

- We can calculate neuron operations in parallel --> GPUs (1000s of cores)
- Helps with generalization
  - instead of a model learning one example at a time, it can learn many at a time
  - this helps prevent the model fit, but too large a batch size (>128) can lead to over-fitting

### Activation Functions

- Every neuron in the hidden layers of a network has an activation function
- **Step Function** Activation Function
  - They will either output 1 or 0
- **Sigmoid**
  - S shape curver with an output between 0 and 1
- **Rectified Linear Unit (ReLU)**

  - if the output > 0, output == ouput
  - if the output <= 0, output == 0
  - elbow shap \_\_/
  - most common

- What is the point of **Activation Functions**?

  - Linear Activation Functions simply can't model curves
    so we need a way to accurately model the data
  - Non-Linear Activation functions are much more precise and efficient
    <br>

    ![linear-activation-function](https://github.com/kawgh1/cat-dog-neural-network/blob/main/images/linear%20activation%20function.png)

- Here is an example of a basic stripped down neural net that is matching it's output
  to a sinusoidal curve
  - the output is only expressed when both neurons in the same row's **activation functions** are active (green)
    at the same time
  - this is a very simple example and not how normal neural nets work, but this provides
    a small window into what is happening under the hood
    <br>
    <br>
    ![neural-net](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/basic%20neural%20net.gif)

### Create our data set

- this dataset will represent three samples (red, blue and green dots) that appear in a spiral formation
  - [here](https://github.com/kawgh1/cat-dog-neural-network/blob/main/create_data.py)
    <br>
    <br>
    - raw data plot
      ![dataset1](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/dataset1.png)
      <br>
    - raw data as 3 color-coded samples (red, blue, green)
      ![dataset1-rgb](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/dataset1%20rgb.png)

### Softmax Activation Function

- In our case, we’re looking to get this model to be a **classifier**, so we want an activation function
  meant for classification. One of these is the **Softmax** activation function. **_First, why are we_**
  **_bothering with another activation function? It just depends on what our overall goals are._**
  - In this case, the rectified linear unit is _unbounded_, _not normalized_ with other units, and _exclusive_.
    - **“Not** **normalized” implies the values can be anything, an output of [12, -99, 318] is without context,**
      **and “exclusive” means each output is independent of the others.**
    - To address this lack of context, the softmax activation on the output data can take in non-normalized, or uncalibrated, inputs and **produce a normalized distribution of probabilities for our classes user Euler's number E, under the hood.**
    - In the case of classification, what we want to see is a prediction of which class the network “thinks” the input represents.
      - This distribution returned by the softmax activation function represents confidence scores for each
        class and will add up to 1. The predicted class is associated with the output neuron that returned
        the largest confidence score. Still, we can also note the other confidence scores in our overarching
        algorithm/program that uses this network.
      - For example, if our classification network has a confidence
        distribution for two classes: **[0.45, 0.55]**, the prediction is the 2nd class, but the confidence in
        this prediction isn’t very high. Maybe our program would not act in this case since it’s not very
        confident.

### Calculating Network Error with Loss Functions

- With a randomly-initialized model, or even a model initialized with more sophisticated
  approaches, our goal is to train, or teach, a model over time. To train a model, we tweak the
  weights and biases to improve the model’s accuracy and confidence. To do this, we calculate how
  much error the model has.
- The loss function, also referred to as the cost function, is the algorithm
  that quantifies how wrong a model is. Loss is the measure of this metric. Since loss is the model’s
  error, we ideally want it to be 0.
  - You may wonder why we do not calculate the error of a model based on the argmax accuracy.
    - Recall our earlier example of confidence: [0.22, 0.6, 0.18] vs [0.32, 0.36, 0.32].
      If the correct class were indeed the middle one (index 1), the model accuracy would be identical
      between the two above. But are these two examples really as accurate as each other? They are
      not, because accuracy is simply applying an argmax to the output to find the index of the biggest
      value. The output of a neural network is actually confidence, and more confidence in the correct
      answer is better. Because of this, we strive to increase correct confidence and decrease misplaced
      confidence.
    ### - Categorical Cross-Entropy Loss
    - Since our model is not a linear regression model, we cannot use the Mean-Squared Error cost function. MSE should only be used on linear regression models.
    - Our model is a **Classification model** so we need a different loss function. The model has a softmax activation function which means it is outputting a probability distribution. **Categorical Cross-Entropy** is explicitly used to compare a "ground-truth" probability (y or "targets") and some predicted distribution (y-hat or "predictions"), so it makes sense to use Cross-Entropy here.
      - **Categorical Cross-Entropy** is one of the most commonly used loss functions with a **softmax activation** on the **output layer**.

### Calculating Cross Entropy Loss Function

- Cross-entropy, also known as logarithmic loss or log loss, is a popular loss function used in machine learning to measure the performance of a classification model. Namely, it measures the difference between the discovered probability distribution of a classification model and the predicted values.

  - Finding the natural log of a number is solving for x:

    - `e ^ x = b`

  - Calculating Cross Entropy Loss

    - ![cross-entropy](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/cross%20entropy.png)
      <br>
    - ![cross-entropy-2](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/cross%20entropy%202.png)
      - Finding Cross Entropy on an output involves finding the Sum &Sigma; of each of the target class value ("1" or "0" when using one-hot encoding) and multiplying it by the log of the predicted value (or the softmax output value).

  - ```
    import math

    softmax_output = [0.7, 0.1, 0.2] # example output from a softmax function

    target_output = [1, 0, 0] # one-hot encoding

    loss = - ( target_output[0] * math.log(softmax_output[0]) +
               target_output[1] * math.log(softmax_output[1]) +
               target_output[2] * math.log(softmax_output[2]) )

    loss = - ( (1 * -0.35667) + (0 * -2.30259) + (0 * -1.60944) )
    loss = - ( -0.35667 + 0 + 0)
    loss = 0.35667
    ```
