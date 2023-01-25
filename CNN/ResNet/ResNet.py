# Residual Networks

# Welcome to the first assignment of this week! You'll be building a very deep convolutional network,
#  using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions;
#  but in practice, they are hard to train.
# Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train
#  much deeper networks than were previously feasible.

# - Implement the basic building blocks of ResNets in a deep neural network using Keras
# - Put together these building blocks to implement and train a state-of-the-art neural
# network for image classification
# - Implement a skip connection in your network


# Before jumping into the problem, run the cell below to load the required packages.

## 1 - Packages

import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
)
from tensorflow.keras.models import Model, load_model
from resnets_utils import *
from tensorflow.keras.initializers import (
    random_uniform,
    glorot_uniform,
    constant,
    identity,
)
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

# from test_utils import summary, comparator
import public_testss
