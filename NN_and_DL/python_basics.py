# Python Basics with Numpy (optional assignment)

# Welcome to your first assignment. This exercise gives you a brief introduction to Python. Even if you've used Python before, this will help familiarize you with the functions we'll need.  

# **Instructions:**
# - You will be using Python 3.
# - Avoid using for-loops and while-loops, unless you are explicitly told to do so.
# - After coding your function, run the cell right below it to check if your result is correct.

## 1 - Building basic functions with numpy ##

# Numpy is the main package for scientific computing in Python. 
# It is maintained by a large community (www.numpy.org). In this exercise you will learn several key numpy functions such as `np.exp`, `np.log`, and `np.reshape`. You will need to know how to use these functions for future assignments.

### 1.1 - sigmoid function, np.exp() ###

# Before using `np.exp()`, you will use `math.exp()` to implement the sigmoid function. 
# You will then see why `np.exp()` is preferable to `math.exp()`.

### Exercise 2 - basic_sigmoid
# Build a function that returns the sigmoid of a real number x. Use `math.exp(x)` for the exponential function.

# **Reminder**:
# $sigmoid(x) = \frac{1}{1+e^{-x}}$ is sometimes also known as the logistic function. 
# It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.
# 
# To refer to a function belonging to a specific package you could call it using `package_name.function()`.
#  Run the code below to see an example with `math.exp()`.


import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x)) 
    
    return s

# Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers.
#  In deep learning we mostly use matrices and vectors. This is why numpy is more useful. 


### One reason why we use "numpy" instead of "math" in Deep Learning ###

# x = [1, 2, 3] # x becomes a python list object
# basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.


import numpy as np

# example of np.exp
t_x = np.array([1, 2, 3])
print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

# Furthermore, if x is a vector, then a Python operation such as $s = x + 3$ or $s = \frac{1}{x}$ 
# will output s as a vector of the same size as x.

# example of vector operation
t_x = np.array([1, 2, 3])
print (t_x + 3)


### Exercise 3 - sigmoid
# Implement the sigmoid function using numpy. 

# **Instructions**: x could now be either a real number, a vector, or a matrix. 
# The data structures we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays.
#  You don't need to know more for now.
# $$ \text{For } x \in \mathbb{R}^n \text{,     } sigmoid(x) = sigmoid\begin{pmatrix}
    # x_1  \\
    # x_2  \\
    # ...  \\
    # x_n  \\
# \end{pmatrix} = \begin{pmatrix}
    # \frac{1}{1+e^{-x_1}}  \\
    # \frac{1}{1+e^{-x_2}}  \\
    # ...  \\
    # \frac{1}{1+e^{-x_n}}  \\
# \end{pmatrix}\tag{1} $$

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """    
    s = 1 / (1 + np.exp(-x))    
    return s

t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

### 1.2 - Sigmoid Gradient

# As you've seen in lecture, you will need to compute gradients to optimize loss functions using backpropagation. 
# Let's code your first gradient function.

### Exercise 4 - sigmoid_derivative
# Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. 
# The formula is: 

# $$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}$$

# You often code this function in two steps:
# 1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
# 2. Compute $\sigma'(x) = s(1-s)$

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    ds = sigmoid(x) * (1 - sigmoid(x))

    return ds

t_x = np.array([1, 2, 3])
print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))


### 1.3 - Reshaping arrays ###

# Two common numpy functions used in deep learning are np.shape and np.reshape(). 
# - X.shape is used to get the shape (dimension) of a matrix/vector X. 
# - X.reshape(...) is used to reshape X into some other dimension. 

# For example, in computer science, an image is represented by a 3D array of shape $(length, height, depth = 3)$.
#  However, when you read an image as the input of an algorithm you convert it to a vector of shape $(length*height*3, 1)$. 
# In other words, you "unroll", or reshape, the 3D array into a 1D vector.

### Exercise 5 - image2vector
# Implement `image2vector()` that takes an input of shape (length, height, 3) and returns a vector of shape (length\*height\*3, 1). For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:
# ``` python
# v = v.reshape((v.shape[0] * v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
# ```
# - Please don't hardcode the dimensions of image as a constant. Instead look up the quantities you need with `image.shape[0]`, etc. 
# - You can use v = v.reshape(-1, 1). Just make sure you understand why it works.

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape(-1,1)
    
    return v

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
t_image = np.array([[[ 0.67826139,  0.29380381],
                     [ 0.90714982,  0.52835647],
                     [ 0.4215251 ,  0.45017551]],

                   [[ 0.92814219,  0.96677647],
                    [ 0.85304703,  0.52351845],
                    [ 0.19981397,  0.27417313]],

                   [[ 0.60659855,  0.00533165],
                    [ 0.10820313,  0.49978937],
                    [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(t_image)))


### 1.4 - Normalizing rows

# Another common technique we use in Machine Learning and Deep Learning is to normalize our data. 
# It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to $ \frac{x}{\| x\|} $ (dividing each row vector of x by its norm).

# For example, if 
# $$x = \begin{bmatrix}
        # 0 & 3 & 4 \\
        # 2 & 6 & 4 \\
# \end{bmatrix}\tag{3}$$ 
# then 
# $$\| x\| = \text{np.linalg.norm(x, axis=1, keepdims=True)} = \begin{bmatrix}
    # 5 \\
    # \sqrt{56} \\
# \end{bmatrix}\tag{4} $$
# and
# $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    # 0 & \frac{3}{5} & \frac{4}{5} \\
    # \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
# \end{bmatrix}\tag{5}$$ 

# Note that you can divide matrices of different sizes and it works fine: this is called broadcasting and you're going to learn about it in part 5.

# With `keepdims=True` the result will broadcast correctly against the original x.

# `axis=1` means you are going to get the norm in a row-wise manner. If you need the norm in a column-wise way, you would need to set `axis=0`. 

# numpy.linalg.norm has another parameter `ord` where we specify the type of normalization to be done (in the exercise below you'll do 2-norm). To get familiar with the types of normalization you can visit [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

### Exercise 6 - normalize_rows
# Implement normalizeRows() to normalize the rows of a matrix. 
# After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).

# **Note**: Don't try to use `x /= x_norm`. For the matrix division numpy must broadcast the x_norm, which is not supported by the operant `/=`

def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """    
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x / x_norm

    return x

x = np.array([[0, 3, 4],
              [1, 6, 4]])
print("normalizeRows(x) = " + str(normalize_rows(x)))


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    #(≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    
    return s


t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(t_x)))

## 2 - Vectorization


# In deep learning, you deal with very large datasets. 
# Hence, a non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run.
# To make sure that your code is  computationally efficient, you will use vectorization. 
# For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.

import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1), len(x2))) # we create a len(x1)*len(x2) matrix with only zeros

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i] * x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")


### Exercise 8 - L1 
# Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.

# **Reminder**:
# - The loss is used to evaluate the performance of your model.
#  The bigger your loss is, the more different your predictions ($ \hat{y} $) are from the true values ($y$). In deep learning, you use optimization algorithms like Gradient Descent to train your model and to minimize the cost.

# - L1 loss is defined as:
# $$\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^{m-1}|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$$

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = np.sum(np.abs(np.subtract(yhat,y)))
   
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    #(≈ 1 line of code)
    loss = np.sum((np.subtract(yhat,y)**2))
    
    return loss