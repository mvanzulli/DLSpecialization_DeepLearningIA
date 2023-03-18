#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Planar data classification with one hidden layer

# Welcome to your week 3 programming assignment!
#  It's time to build your first neural network, which will have one hidden layer. 
# Now, you'll notice a big difference between this model and the one you implemented 
# previously using logistic regression.

# By the end of this assignment, you'll be able to:

# - Implement a 2-class classification neural network with a single hidden layer
# - Use units with a non-linear activation function, such as tanh
# - Compute the cross entropy loss
# - Implement forward and backward propagation

# 1 - Packages

# First import all the packages that you will need during this assignment.

# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis. 
# - [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
# - testCases provides some test examples to assess the correctness of your functions
# - planar_utils provide various useful functions used in this assignment

import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# 2 - Load the Dataset 

X, Y = load_planar_dataset()

# Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) 
# and some blue (y=1) points. Your goal is to build a model to fit this data.
#  In other words, we want the classifier to define regions as either red or blue.


# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()


shape_X = X.shape
shape_Y = Y.shape
m = shape_X[0]

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

## 3 - Simple Logistic Regression

# Before building a full neural network, let's check how logistic regression performs on this problem. 
# You can use sklearn's built-in functions for this. Run the code below to train a logistic regression classifier on the dataset.


# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)


# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y);
plt.title("Logistic Regression");

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
plt.show()

# **Interpretation**: The dataset is not linearly separable, so logistic regression doesn't perform well. 
# Hopefully a neural network will do better. Let's try this now! 

## 4 - Neural Network model

# Logistic regression didn't work well on the flower dataset. 
# Next, you're going to train a Neural Network with a single hidden layer and see how that handles the same problem.



# **Reminder**: The general methodology to build a Neural Network is to:
    # 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
    # 2. Initialize the model's parameters
    # 3. Loop:
        # - Implement forward propagation
        # - Compute loss
        # - Implement backward propagation to get the gradients
        # - Update parameters (gradient descent)

# In practice, you'll often build helper functions to compute steps 1-3, then merge them into one function called `nn_model()`.
#  Once you've built `nn_model()` and learned the right parameters, you can make predictions on new data.


### 4.1 - Defining the neural network structure ####

### Exercise 2 - layer_sizes 

# Define three variables:
    # - n_x: the size of the input layer
    # - n_h: the size of the hidden layer (**set this to 4, only for this Exercise 2**) 
    # - n_y: the size of the output layer
# 
# **Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

### 4.2 - Initialize the model's parameters ####

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0] 
        
    return (n_x, n_h, n_y)

### Exercise 3 -  initialize_parameters
t_X, t_Y = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(t_X, t_Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

# layer_sizes_test(layer_sizes)


# Implement the function `initialize_parameters()`.

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

np.random.seed(2)
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test(initialize_parameters)


### 4.3 - The Loop 

### Exercise 4 - forward_propagation

# Implement `forward_propagation()` using the following equations:

# $$Z^{[1]} =  W^{[1]} X + b^{[1]}\tag{1}$$ 
# $$A^{[1]} = \tanh(Z^{[1]})\tag{2}$$
# $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}\tag{3}$$
# $$\hat{Y} = A^{[2]} = \sigma(Z^{[2]})\tag{4}$$


# - Check the mathematical representation of your classifier in the figure above.
# - Use the function `sigmoid()`. It's built into (imported) this notebook.
# - Use the function `np.tanh()`. It's part of the numpy library.
# - Implement using these steps:
#     1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()` 
# by using `parameters[".."]`.
#     2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
# - Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # In the general case W = nneurons, nfeatures and X has nfeatures nsamples   

    # Implement Forward Propagation to calculate A2 (probabilities)    # (≈ 4 lines of code)
    print(W1.shape)
    print(X.shape)

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

t_X, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(t_X, parameters)
print("A2 = " + str(A2))

forward_propagation_test(forward_propagation)