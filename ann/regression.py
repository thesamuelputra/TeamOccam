import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

# Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 97
n_output = 2

# Learning parameters
learning_constant = 0.2
number_epochs = 1000
batch_size = 1000

# Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
# DEFINING WEIGHTS AND BIASES
# Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
# Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
# Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))
# Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
# Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
# Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    # Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3), b3)

    return out_layer


# Create model
neural_network = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
# loss_op =
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=neural_network, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(
    learning_constant).minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

x = pd.read_csv('./datasets/CommViolPredUnnormalizedData.txt', sep=",",
                header=None, usecols=[*range(5, 103)])
x = x.drop(columns=(30), axis=1)
y = pd.read_csv('./datasets/CommViolPredUnnormalizedData.txt', sep=",",
                header=None, usecols=[*range(129, 131)])

# print(x, y)

label = y  # +1e-50-1e-50
batch_x_train = x.sample(frac=0.7, random_state=0)
batch_y_train = y.sample(frac=0.7, random_state=0)
batch_x_test = x.drop(batch_x_train.index)
batch_y_test = x.drop(batch_y_train.index)
# label_train = label[0:599]
# label_test = label[600:1000]


def get_normalization_parameters(traindf, features):
    def _z_score_params(column):
        mean = traindf[column].mean()
        std = traindf[column].std()
        return {'col': column, 'mean': mean, 'std': std}

    normalization_parameters = []
    for column in features:
        normalization_parameters.append(_z_score_params(column))
    return normalization_parameters


def normalize_column(col, mean, std):  # Use mean, std defined below.
    return (col - mean)/std


def create_feature_cols(features, normalization_parameters):
    normalized_x_train = []
    for i in range(len(normalization_parameters)):
        x = normalize_column(
            features.values[:, i], normalization_parameters[i]['mean'], normalization_parameters[i]['std'])
        normalized_x_train.append(x)
    return normalized_x_train


normalization_x_parameters = get_normalization_parameters(
    batch_x_train, x)
normalized_x_train = create_feature_cols(
    batch_x_train, normalization_x_parameters)
normalized_x_train = pd.DataFrame(np.transpose(normalized_x_train))

normalization_y_parameters = get_normalization_parameters(
    batch_y_train, y)
normalized_y_train = create_feature_cols(
    batch_y_train, normalization_y_parameters)
normalized_y_train = pd.DataFrame(np.transpose(normalized_y_train))

# print(batch_y_train)
# print(normalization_y_parameters)
# print(normalized_y_train)

# print(normalized_x_train)
# print(batch_x_train)
# print(normalized_y_train)
# print(batch_y_train)


with tf.Session() as sess:
    sess.run(init)
    # Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: normalized_x_train, Y:
                                       normalized_y_train})
        # Display the epoch
        if epoch % 100 == 0:
            print("Epoch:", '%d' % (epoch))
    # Test model
    pred = (neural_network)  # Apply softmax to logits
    accuracy = tf.keras.losses.MSE(pred, Y)

    print("Accuracy:", accuracy.eval({X: normalized_x_train, Y:
                                      normalized_y_train}))
    # tf.keras.evaluate(pred,batch_x)
    print("Prediction:", pred.eval({X: normalized_x_train}))
    output = neural_network.eval({X: normalized_x_train})
    plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'bo')
    plt.ylabel('some numbers')
    plt.show()

    estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
    correct_prediction1 = tf.equal(
        tf.argmax(pred, 1), tf.argmax(normalized_y_train, 1))  # error
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    print("Accuracy1", accuracy1.eval({X: normalized_x_train}))
