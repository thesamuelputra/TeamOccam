import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

n_hidden1 = 50
n_hidden2 = 25
n_input = 97
n_output = 2
# learning parameters
number_epochs = 1000

def plot_loss(history):
    plt.figure('loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

def plot_accuracy(history):
    plt.figure('accuracy')
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

x = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(5, 103)])
x = x.drop(columns=(30), axis=1)
y = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(129, 131)])
batch_x_train = x.sample(frac=0.7, random_state=0)
batch_y_train = y.sample(frac=0.7, random_state=0)
batch_x_test = x.drop(batch_x_train.index)
batch_y_test = y.drop(batch_y_train.index)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='sigmoid'),
    tf.keras.layers.Dense(n_hidden2, activation='sigmoid'),
    tf.keras.layers.Dense(n_output, activation='linear')
])

model.compile(loss='mean_absolute_error', optimizer='sgd',metrics=["accuracy"])
history = model.fit(batch_x_train, batch_y_train, epochs=number_epochs, validation_split=0.2)

test_result = model.evaluate(batch_x_test, batch_y_test)

print(test_result)

plot_loss(history)
plot_accuracy(history)
plt.show()