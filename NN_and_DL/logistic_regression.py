# Logistic Regression with a Neural Network mindset

# Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize  cats. This assignment will step you through how to do this with a Neural Network mindset, and will also hone your intuitions about deep learning.

# - Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
# - Use `np.dot(X,Y)` to calculate dot products.

# - Build the general architecture of a learning algorithm, including:
    # - Initializing parameters
    # - Calculating the cost function and its gradient
    # - Using an optimization algorithm (gradient descent) 
# - Gather all three functions above into a main model function, in the right order.

## 1 - Packages ##

# First, let's run the cell below to import all the packages that you will need during this assignment. 
# - [numpy](https://numpy.org/doc/1.20/) is the fundamental package for scientific computing with Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
# - [PIL](https://pillow.readthedocs.io/en/stable/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
## 2 - Overview of the Problem set ##

    # - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
    # - a test set of m_test images labeled as cat or non-cat
    # - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). 
    # Thus, each image is square (height = num_px) and (width = num_px).

# You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

# Let's get more familiar with the dataset. Load the data by running the following code.


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. 
# After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).

# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. 
# Feel free also to change the `index` value and re-run to see other images. 

# Example of a picture
index = 60
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs. 

### Exercise 1
# Find the values for:
    # - m_train (number of training examples)
    # - m_test (number of test examples)
    # - num_px (= height = width of a training image)
# Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). 
# For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`.

m_train = train_set_x_orig.shape[0] 
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# For convenience, you should now reshape images of shape (num_px, num_px, 3) 
# in a numpy-array of shape (num_px $*$ num_px $*$ 3, 1). After this, our training (and test) 
# dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.

### Exercise 2
# Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape 
# (num\_px $*$ num\_px $*$ 3, 1).

# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: 
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

# Reshape the training and test examples
#(≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

# Check that the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, 
# and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
# 
# One common preprocessing step in machine learning is to center and standardize your dataset, 
# meaning that you substract the mean of the whole numpy array from each example, 
# and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, 
# it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

#  During the training of your model, you're going to multiply weights and add biases to some initial inputs in order to observe neuron activations. 
# Then you backpropogate with the gradients to train the model. But, it is extremely important for each feature to have a similar range such that our gradients don't explode.
#  You will see that more in detail later in the lectures. !--> 

# Let's standardize our dataset.

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# **What you need to remember:**

# Common steps for pre-processing a new dataset are:
# - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# - Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
# - "Standardize" the data

## 3 - General Architecture of the learning algorithm ##

# It's time to design a simple algorithm to distinguish cat images from non-cat images.

# You will build a Logistic Regression, using a Neural Network mindset. 
# The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**

# **Mathematical expression of the algorithm**:

# For one example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

# **Key steps**:
# In this exercise, you will carry out the following steps: 
    # - Initialize the parameters of the model
    # - Learn the parameters for the model by minimizing the cost  
    # - Use the learned parameters to make predictions (on the test set)
    # - Analyse the results and conclude

## 4 - Building the parts of our algorithm ## 

# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
    # - Calculate current loss (forward propagation)
    # - Calculate current gradient (backward propagation)
    # - Update parameters (gradient descent)

# You often build 1-3 separately and integrate them into one function we call `model()`.

### Exercise 3 - sigmoid
# Using your code from "Python Basics", implement `sigmoid()`. 
# As you've seen in the figure above, you need to compute $sigmoid(z) = \frac{1}{1 + e^{-z}}$ for $z = w^T x + b$ to make predictions. 
# Use np.exp().

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    
    return s

### Exercise 4 - initialize_with_zeros
# Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros. 
# If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    
    w = np.zeros((dim,1))
    b = 0.
    
    return w, b

### 4.3 - Forward and Backward propagation

# Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.

### Exercise 5 - propagate
# Implement a function `propagate()` that computes the cost function and its gradient.

# Forward Propagation:
# - You get X
# - You compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
# - You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$

# Here are the two formulas you will be using: 

# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    #(≈ 2 lines of code)
    # compute activation
    A = sigmoid(np.dot(w.T, X) + b)
    # And don't use loops for the sum.
    cost = -1/m * np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A))                                

    # BACKWARD PROPAGATION (TO FIND GRAD)
    #(≈ 2 lines of code)
    dw = 1/m * np.matmul(X,np.subtract(A,Y).T)
    db = 1/m * np.sum(A - Y)
   
    # YOUR CODE ENDS HERE
    cost = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

w =  np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])
grads, cost = propagate(w, b, X, Y)

assert type(grads["dw"]) == np.ndarray
assert grads["dw"].shape == (2, 1)
assert type(grads["db"]) == np.float64
    

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

# **Expected output**

# dw = [[ 0.25071532]
#  [-0.06604096]]
# db = -0.1250040450043965
# cost = 0.15900537707692405
propagate_test(propagate)
print("Gradients are computed correctly")
### 4.4 - Optimization
# - You have initialized your parameters.
# - You are also able to compute a cost function and its gradient.
# - Now, you want to update the parameters using gradient descent.

### Exercise 6 - optimize
# Write down the optimization function. 
# The goal is to learn $w$ and $b$ by minimizing the cost function $J$. 
# For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.
# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # (≈ 1 lines of code)
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("Costs = " + str(costs))

### Exercise 7 - predict
# The previous function will output the learned w and b. 
# We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. 
# There are two steps to computing predictions:

# 1. Calculate $\hat{Y} = A = \sigma(w^T X + b)$

# 2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. 
# If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.matmul(w.T,X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > .5 :
            Y_prediction[0,i] = 1 
        else:
            Y_prediction[0,i] = 0 
    
    return Y_prediction

w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
print ("predictions = " + str(predict(w, b, X)))

## 5 - Merge all functions into a model ##

# You will now see how the overall model is structured by putting together all the building blocks 
# (functions implemented in the previous parts) together, in the right order.

### Exercise 8 - model
# Implement the model function. Use the following notation:
    # - Y_prediction_test for your predictions on the test set
    # - Y_prediction_train for your predictions on the train set
    # - parameters, grads, costs for the outputs of optimize()


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros 
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)
    
    # Gradient descent 
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate,print_cost)
    
    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

model_test(model)