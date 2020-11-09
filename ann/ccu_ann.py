import numpy as np
import tensorflow as tf
import pandas as pd
# import logging
# logging.basicConfig(level=logging.DEBUG)
# import matplotlib.pyplot as plt

n_hidden1 = 36
n_hidden2 = 36
n_input = 97
n_output = 2
# learning parameters
learning_constants = 0.2
number_epochs = 1000

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

# biases
b1 = tf.Variable(tf.random.normal([n_hidden1]))
b2 = tf.Variable(tf.random.normal([n_hidden2]))
b3 = tf.Variable(tf.random.normal([n_output]))

# weights
w1 = tf.Variable(tf.random.normal([n_input, n_hidden1]))
w2 = tf.Variable(tf.random.normal([n_hidden1, n_hidden2]))
w3 = tf.Variable(tf.random.normal([n_hidden2, n_output]))

def multilayer_perceptron(input_d):
    input_d_norm = tf.layers.batch_normalization(input_d)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d_norm, w1), b1))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    out_layer = tf.add(tf.matmul(layer_2, w3), b3)

    return out_layer

neural_network = multilayer_perceptron(X)

# loss function and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constants).minimize(loss_op)

init = tf.global_variables_initializer()

x = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(5, 103)])
x = x.drop(columns=(30), axis=1)
y = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(129, 131)])
x_train = x.sample(frac=0.7, random_state=0)
y_train = y.sample(frac=0.7, random_state=0)

correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(neural_network, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(number_epochs + 1):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
        accuracy_value = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
        loss_value = sess.run(loss_op, feed_dict={X: x_train, Y: y_train})
        print("Epoch: {} - loss: {}, accuracy: {}".format(epoch, loss_value, accuracy_value))
        # if epoch % 100 == 0:
        #     print("Epoch: {}".format(epoch))
    # pred = (neural_network)
    # print(pred)
    # accuracy = tf.keras.losses.MSE(pred, Y)
    # print("Accuracy: {}".format(accuracy.eval({X: batch_x_train, Y: batch_y_train}).size))