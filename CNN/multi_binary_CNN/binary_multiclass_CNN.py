# Convolutional Neural Networks: Application

# Welcome to Course 4's second assignment! In this notebook, you will:

# - Create a mood classifer using the TF Keras Sequential API
# - Build a ConvNet to identify sign language digits using the TF Keras Functional API

# - Build and train a ConvNet in TensorFlow for a __binary__ classification problem
# - Explain different use cases for the Sequential and Functional APIs

# To complete this assignment, you should already be familiar with TensorFlow.

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

### 1.1 - Load the Data and Split the Data into Train/Test Sets

# You'll be using the Happy House dataset for this part of the assignment,
# which contains images of peoples' faces. Your task will be to build a
# ConvNet that determines whether the people in the images are smiling or
# not -- because they only get to enter the house if they're smiling!

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

index = 50
plt.imshow(X_train_orig[index])  # display sample training image
plt.show()

## 2 - Layers in TF Keras

# In the previous assignment, you created layers manually in numpy.
# In TF Keras, you don't have to write code directly to create layers.
# Rather, TF Keras has pre-defined layers you can use.

# When you create a layer in TF Keras, you are creating a function that takes
#  some input and transforms it into an output you can reuse later. Nice and easy!

## 3 - The Sequential API

# In the previous assignment, you built helper functions using `numpy` to understand
# the mechanics behind convolutional neural networks. Most practical applications of
# deep learning today are built using programming frameworks, which have many built-in
# functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow,
# which allows for even more simplified and optimized model creation and training.

# For the first part of this assignment, you'll create a model using TF Keras' Sequential API,
#  which allows you to build layer by layer, and is ideal for building models where each layer
#  has **exactly one** input tensor and **one** output tensor.

# As you'll see, using the Sequential API is simple and straightforward, but is only appropriate
# for simpler, more straightforward tasks. Later in this notebook you'll spend some time building
# with a more flexible, powerful alternative: the Functional API.


### 3.1 - Create the Sequential Model

# As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with
# layer operations that proceed in a sequential order.

# You can also add layers incrementally to a Sequential model with the `.add()` method, or remove
# them using the `.pop()` method, much like you would in a regular Python list.

# Actually, you can think of a Sequential model as behaving like a list of layers.
# Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.
# If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential
# model wouldn't be the right choice!

# For any layer construction in Keras, you'll need to specify the input shape in advance.
# This is because in Keras, the shape of the weights is based on the shape of the inputs.
# The weights are only created when the model first sees some input data. Sequential models can be
# created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

### Exercise 1 - happyModel

# Implement the `happyModel` function below to build the following model:
# `ZEROPAD2D ->
#  CONV2D ->
#  BATCHNORM ->
#  RELU ->
#  MAXPOOL ->
#  FLATTEN ->
#  DENSE`.
#
#  Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)

# Also, plug in the following parameters for all the steps:

#  - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
#  - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
#  - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
#  - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
#  - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
#  - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
#  - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer:
# Apply a fully connected layer with 1 neuron and a sigmoid activation.


def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = tf.keras.Sequential(
        [
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tf.keras.layers.ZeroPadding2D(input_shape=(64, 64, 3), padding=(3, 3)),
            ## Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1)),
            ## BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis=3),
            ## ReLU
            tf.keras.layers.ReLU(),
            ## Max Pooling 2D with default parameters
            tf.keras.layers.MaxPool2D(),
            ## Flatten layer
            tf.keras.layers.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    return model


# Test the model
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)

output = [
    ["ZeroPadding2D", (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
    ["Conv2D", (None, 64, 64, 32), 4736, "valid", "linear", "GlorotUniform"],
    ["BatchNormalization", (None, 64, 64, 32), 128],
    ["ReLU", (None, 64, 64, 32), 0],
    ["MaxPooling2D", (None, 32, 32, 32), 0, (2, 2), (2, 2), "valid"],
    ["Flatten", (None, 32768), 0],
    ["Dense", (None, 1), 32769, "sigmoid"],
]

comparator(summary(happy_model), output)


# Now that your model is created, you can compile it for training with an optimizer and loss of your choice.
#  When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically
# converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier!

happy_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# It's time to check your model's parameters with the `.summary()` method.
# This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer.

happy_model.summary()

# 3.2 - Train and Evaluate the Model
# After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents,
#  you are now ready to build!

# Simply call .fit() to train. That's it! No need for mini-batching, saving, or complex backpropagation computations.
# That's all been done for you, as you're using a TensorFlow dataset with the batches specified already.
# You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).

happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)


# After that completes, just use `.evaluate()` to evaluate against your test set.
#  This function will print the value of the loss function and the performance metrics
# specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.

happy_model.evaluate(X_test, Y_test)


## 4 - The Functional API

# Welcome to the second half of the assignment, where you'll use Keras' flexible
# [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that
# can differentiate between 6 sign language digits.

# The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs.
# Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows
# much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect
# in many more ways than one.

# In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection,
# which is just one of the many ways a Functional model can be constructed.
# A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network.
# Don't worry, you'll be spending more time with skip connections very soon!


### 4.1 - Load the SIGNS Dataset

# As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()

# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:, index])))

### 4.2 - Split the Data into Train/Test Sets

# In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset,
# it is more natural to apply a ConvNet to it.

# To get started, let's examine the shapes of your data.

X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


### 4.3 - Forward Propagation

# In TensorFlow, there are built-in functions that implement the convolution steps for you.
#  By now, you should be familiar with how TensorFlow builds computational graphs.
# In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers.
#  This is what allows such great flexibility.

# However, the following model could also be defined using the Sequential API since the information flow
# is on a single line. But don't deviate. What we want you to learn is to use the functional API.

# Begin building your graph of layers by creating an input node that functions as a callable object:

### Exercise 2 - convolutional_model

# Implement the `convolutional_model` function below to build the following model:
# `CONV2D ->
#  RELU ->
#  MAXPOOL ->
#  CONV2D ->
#  RELU ->
#  MAXPOOL ->
#  FLATTEN ->
#  DENSE`.

# Also, plug in the following parameters for all the steps:

#  - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
#  - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
#  - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride,
#  padding is "SAME"
#  - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
#  - **ReLU**
#  - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
#  - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
#  - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer:
#  Apply a fully connected layer with 6 neurons and a softmax activation.


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape, name="InputLayer")
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(8, (4, 4), strides=(1, 1), padding="same", name="Conv2D_1")(
        input_img
    )
    ## RELU
    A1 = tfl.ReLU(name="ReLu_1")(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(
        pool_size=(8, 8), strides=(8, 8), padding="same", name="MaxPooling2D_1"
    )(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(16, (2, 2), strides=(1, 1), padding="same", name="Conv2D")(P1)
    ## RELU
    A2 = tfl.ReLU(name="ReLu_2")(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P1 = tfl.MaxPool2D(
        pool_size=(4, 4), strides=(4, 4), padding="same", name="MaxPooling2D_2"
    )(A2)
    ## FLATTEN
    F = tfl.Flatten(name="Flatten")(P1)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    outputs = tf.keras.layers.Dense(6, activation="softmax", name="Dense")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


conv_model = convolutional_model((64, 64, 3))
conv_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
conv_model.summary()

output = [
    ["InputLayer", [(None, 64, 64, 3)], 0],
    ["Conv2D", (None, 64, 64, 8), 392, "same", "linear", "GlorotUniform"],
    ["ReLU", (None, 64, 64, 8), 0],
    ["MaxPooling2D", (None, 8, 8, 8), 0, (8, 8), (8, 8), "same"],
    ["Conv2D", (None, 8, 8, 16), 528, "same", "linear", "GlorotUniform"],
    ["ReLU", (None, 8, 8, 16), 0],
    ["MaxPooling2D", (None, 2, 2, 16), 0, (4, 4), (4, 4), "same"],
    ["Flatten", (None, 64), 0],
    ["Dense", (None, 6), 390, "softmax"],
]

comparator(summary(conv_model), output)

### 4.4 - Train the Model
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)


## 5 - History Object

# The history object is an output of the `.fit()` operation, and provides a record
# of all the loss and metric values in memory. It's stored as a dictionary that you
#  can retrieve at `history.history`:
#
print(history.history)

## 6 Plotting the Loss and Accuracy

# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on.
df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[["loss", "val_loss"]]
df_loss.rename(columns={"loss": "train", "val_loss": "validation"}, inplace=True)
df_acc = df_loss_acc[["accuracy", "val_accuracy"]]
df_acc.rename(columns={"accuracy": "train", "val_accuracy": "validation"}, inplace=True)
df_loss.plot(title="Model loss", figsize=(12, 8)).set(xlabel="Epoch", ylabel="Loss")
df_acc.plot(title="Model Accuracy", figsize=(12, 8)).set(
    xlabel="Epoch", ylabel="Accuracy"
)

plt.show()
