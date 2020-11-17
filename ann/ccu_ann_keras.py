import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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

def plot_cv_rmse(average_rmse):
    plt.figure('cv_rmse')
    plt.plot(average_rmse, label='rmse')
    plt.plot(average_rmse, 'ro')
    plt.xlabel('folds')
    plt.ylabel('rmse')
    plt.legend()

def plot_cv_accuracy(average_accuracy):
    plt.figure('cv_accuracy')
    plt.plot(average_accuracy, label='accuracy')
    plt.plot(average_accuracy, 'ro')
    plt.ylim([0, 1])
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.legend()

# load datasets 
x = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(5, 103)])
x = x.drop(columns=(30), axis=1)
y = pd.read_csv("datasets/ccu_no_header.csv", sep=",", header=None, usecols=[*range(129, 131)])
batch_x_train = x.sample(frac=0.7, random_state=0)
batch_y_train = y.sample(frac=0.7, random_state=0)
batch_x_test = x.drop(batch_x_train.index)
batch_y_test = y.drop(batch_y_train.index)

#normalize datasets
normalization_x_parameters = get_normalization_parameters(
    x, x)
normalized_x = create_feature_cols(
    x, normalization_x_parameters)
normalized_x = np.transpose(normalized_x)

normalization_y_parameters = get_normalization_parameters(
    y, y)
normalized_y = create_feature_cols(
    y, normalization_y_parameters)
normalized_y = np.transpose(normalized_y)

""" # normalize train datasets
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

# normalize test datasets
normalization_x_test_parameters = get_normalization_parameters(
    batch_x_test, x)
normalized_x_test = create_feature_cols(
    batch_x_test, normalization_x_test_parameters)
normalized_x_test = pd.DataFrame(np.transpose(normalized_x_test))

normalization_y_test_parameters = get_normalization_parameters(
    batch_y_test, y)
normalized_y_test = create_feature_cols(
    batch_y_test, normalization_y_test_parameters)
normalized_y_test = pd.DataFrame(np.transpose(normalized_y_test)) """

# multilayer perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='sigmoid'),
    tf.keras.layers.Dense(n_hidden2, activation='sigmoid'),
    tf.keras.layers.Dense(n_output, activation='linear')
])
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=["accuracy", tf.keras.metrics.RootMeanSquaredError()])

kf = KFold(n_splits=10)

k_fold = 1
cv_accuracy_scores = []
cv_rmse_scores = []
for train_index, test_index in kf.split(x):
    print("Cross fold validation - fold {}".format(k_fold))
    x_train, x_test = normalized_x[train_index], normalized_x[test_index]
    y_train, y_test = normalized_y[train_index], normalized_y[test_index]
    model.fit(x_train, y_train, epochs=100, verbose=1)
    loss, accuracy, rmse = model.evaluate(x_test, y_test)
    cv_accuracy_scores.append(accuracy)
    cv_rmse_scores.append(rmse)
    
    k_fold += 1

print("-" * 70)
print("Average accuracy: {}".format(np.mean(cv_accuracy_scores)))
print("Average root mean square error: {}".format(np.mean(cv_rmse_scores)))

plot_cv_accuracy(cv_accuracy_scores)
plot_cv_rmse(cv_rmse_scores)
plt.show()

# plt.scatter(normalized_x_train.to_numpy()[:,5], normalized_y_train.to_numpy()[:,0])
# plt.title('ccu visualization')
# plt.xlabel('features')
# plt.ylabel('labels/targets')
# plt.show()

# model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=["accuracy"])
# history = model.fit(normalized_x_train, normalized_y_train, epochs=number_epochs, validation_split=0.2)

# loss, accuracy = model.evaluate(normalized_x_test, normalized_y_test)

# print("Test result: loss ({}), accuracy ({})".format(loss, accuracy))

# plot_loss(history)
# plot_accuracy(history)
# plt.show()