# Convolutional Neural Networks: Step by Step

# Welcome to Course 4's first assignment! In this assignment,
#  you will implement convolutional (CONV) and pooling (POOL)
#  layers in numpy, including both forward propagation and
# (optionally) backward propagation.

# By the end of this notebook, you'll be able to:

# * Explain the convolution operation
# * Apply two different types of pooling operation
# * Identify the components used in a convolutional
# neural network (padding, stride, filter, ...) and
# their purpose
# * Build a convolutional neural network


# **Notation**:
# - Superscript $[l]$ denotes an object of the $l^{th}$ layer.
# - Example: $a^{[4]}$ is the $4^{th}$ layer activation.
# $W^{[5]}$ and $b^{[5]}$ are the $5^{th}$ layer parameters.


# - Superscript $(i)$ denotes an object from the $i^{th}$ example.
# - Example: $x^{(i)}$ is the $i^{th}$ training example input.


# - Subscript $i$ denotes the $i^{th}$ entry of a vector.
# - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the
# activations in layer $l$, assuming this is a fully connected
#  (FC) layer.


# - $n_H$, $n_W$ and $n_C$ denote respectively the height, width
# and number of channels of a given layer. If you want to reference
#  a specific layer $l$, you can also write $n_H^{[l]}$, $n_W^{[l]}$,
# $n_C^{[l]}$.
# - $n_{H_{prev}}$, $n_{W_{prev}}$ and $n_{C_{prev}}$ denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer $l$, this could also be denoted $n_H^{[l-1]}$, $n_W^{[l-1]}$, $n_C^{[l-1]}$.

## 1 - Packages

import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *

plt.rcParams["figure.figsize"] = (5.0, 4.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)

## 2 - Outline of the Assignment

# You will be implementing the building blocks
# of a convolutional neural network! Each function
#  you will implement will have detailed instructions
#   to walk you through the steps:

# - Convolution functions, including:
#     - Zero Padding
#     - Convolve window
#     - Convolution forward
#     - Convolution backward (optional)
# - Pooling functions, including:
#     - Pooling forward
#     - Create mask
#     - Distribute value
#     - Pooling backward (optional)


## 3 - Convolutional Neural Networks

# Although programming frameworks make convolutions easy to use,
# they remain one of the hardest concepts to understand in Deep Learning.
# A convolution layer transforms an input volume into an output volume
# of different size, as shown below.


### 3.1 - Zero-Padding

# Zero-padding adds zeros around the border of an image:

# The main benefits of padding are:

# - It allows you to use a CONV layer without necessarily shrinking
#  the height and width of the volumes. This is important for building
#   deeper networks, since otherwise the height/width would shrink as
#   you go to deeper layers. An important special case is the "same"
#   convolution, in which the height/width is exactly preserved after
#   one layer.

# - It helps us keep more of the information at the border of an image.
# Without padding, very few values at the next layer would be affected
#  by pixels at the edges of an image.

### Exercise 1 - zero_pad
# Implement the following function, which pads all the images of a batch
# of examples X with zeros. [Use np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html).
# Note if you want to pad the array "a" of shape $(5,5,5,5,5)$ with `pad = 1`
# for the 2nd dimension, `pad = 3` for the 4th dimension and `pad = 0` for the rest,
# you would do:
# ```python
# a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), mode='constant', constant_values = (0,0))
# ```


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the
    height and width of an image, as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))

    return X_pad


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print("x.shape =\n", x.shape)
print("x_pad.shape =\n", x_pad.shape)
print("x[1,1] =\n", x[1, 1])
print("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title("x")
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title("x_pad")
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)


### 3.2 - Single Step of Convolution 

# In this part, implement a single step of convolution, in which you apply the filter
#  to a single position of the input. This will be used to build a convolutional unit, 
# which: 

# - Takes an input volume 
# - Applies a filter at every position of the input
# - Outputs another volume (usually of different size)

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = z + float(b)

    return Z

    np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"