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




