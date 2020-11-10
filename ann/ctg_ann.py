import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

n_hidden1 = 50
n_hidden2 = 25
n_input = 21
n_output = 10
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
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

x = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='K:AE')
y = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='AR')
y = y - 1
# y = to_categorical(y, n_output)

batch_x_train = x.sample(frac=0.7, random_state=0)
batch_y_train = y.sample(frac=0.7, random_state=0)
batch_x_test = x.drop(batch_x_train.index)
batch_y_test = y.drop(batch_y_train.index)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='sigmoid'),
    tf.keras.layers.Dense(n_hidden2, activation='sigmoid'),
    tf.keras.layers.Dense(n_output, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
history = model.fit(batch_x_train, batch_y_train, epochs=number_epochs, validation_split=0.2)

loss, accuracy = model.evaluate(batch_x_test, batch_y_test)

print("Test result: loss ({}), accuracy ({})".format(loss, accuracy))

plot_loss(history)
plot_accuracy(history)
plt.show()