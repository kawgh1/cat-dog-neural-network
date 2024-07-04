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
    ![dataset1](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/dataset1.png)
    ![dataset1-rgb](https://raw.githubusercontent.com/kawgh1/cat-dog-neural-network/main/images/dataset1%20rgb.png)
