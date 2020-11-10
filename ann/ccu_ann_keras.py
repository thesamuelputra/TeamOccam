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

def plot_loss(history):
    plt.figure('loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='sigmoid'),
    tf.keras.layers.Dense(n_hidden2, activation='sigmoid'),
    tf.keras.layers.Dense(n_output, activation='linear')
])

model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=["accuracy"])
history = model.fit(normalized_x_train, normalized_y_train, epochs=number_epochs, validation_split=0.2)

normalization_x_test_parameters = get_normalization_parameters(
    batch_x_test, x)
normalized_x_test = create_feature_cols(
    batch_x_test, normalization_x_test_parameters)
normalized_x_test = pd.DataFrame(np.transpose(normalized_x_test))

normalization_y_test_parameters = get_normalization_parameters(
    batch_y_test, y)
normalized_y_test = create_feature_cols(
    batch_y_test, normalization_y_test_parameters)
normalized_y_test = pd.DataFrame(np.transpose(normalized_y_test))

loss, accuracy = model.evaluate(normalized_x_test, normalized_y_test)

print("Test result: loss ({}), accuracy ({})".format(loss, accuracy))

plot_loss(history)
plot_accuracy(history)
plt.show()