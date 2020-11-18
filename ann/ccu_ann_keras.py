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
number_epochs = 100

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
batch_x_train = x.sample(frac=0.8, random_state=0)
batch_y_train = y.sample(frac=0.8, random_state=0)
batch_x_test = x.drop(batch_x_train.index)
batch_y_test = y.drop(batch_y_train.index)

# normalize train datasets
normalization_x_parameters = get_normalization_parameters(
    batch_x_train, x)
normalized_x_train = create_feature_cols(
    batch_x_train, normalization_x_parameters)
normalized_x_train = np.transpose(normalized_x_train)

normalization_y_parameters = get_normalization_parameters(
    batch_y_train, y)
normalized_y_train = create_feature_cols(
    batch_y_train, normalization_y_parameters)
normalized_y_train = np.transpose(normalized_y_train)

# normalize test datasets
normalization_x_test_parameters = get_normalization_parameters(
    batch_x_test, x)
normalized_x_test = create_feature_cols(
    batch_x_test, normalization_x_test_parameters)
normalized_x_test = np.transpose(normalized_x_test)

normalization_y_test_parameters = get_normalization_parameters(
    batch_y_test, y)
normalized_y_test = create_feature_cols(
    batch_y_test, normalization_y_test_parameters)
normalized_y_test = np.transpose(normalized_y_test)

# multilayer perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='tanh'),
    tf.keras.layers.Dense(n_hidden2, activation='relu'),
    tf.keras.layers.Dense(n_output, activation='linear')
])
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=["accuracy", tf.keras.metrics.RootMeanSquaredError()])

kf = KFold(n_splits=10)

k_fold = 1
cv_accuracy_scores = []
cv_rmse_scores = []
for train_index, test_index in kf.split(normalized_x_train):
    print("Cross fold validation - fold {}".format(k_fold))
    x_train, x_test = normalized_x_train[train_index], normalized_x_train[test_index]
    y_train, y_test = normalized_y_train[train_index], normalized_y_train[test_index]
    model.fit(x_train, y_train, epochs=number_epochs, verbose=1)
    loss, accuracy, rmse = model.evaluate(x_test, y_test)
    cv_accuracy_scores.append(accuracy)
    cv_rmse_scores.append(rmse)

    k_fold += 1

saved_model_path = "./saved_models/regression_model_occam.h5"
model.save(saved_model_path)
print("MODEL SAVED!!!")

print("-" * 70)
print("Average accuracy: {} (+- {})".format(np.mean(cv_accuracy_scores), np.std(cv_accuracy_scores)))
print("Average root mean square error: {}".format(np.mean(cv_rmse_scores)))

predicted = model.predict(normalized_x_test)

fig, ax = plt.subplots()
ax.scatter(normalized_y_test[:,1], predicted[:,1], edgecolors=(0, 0, 0), color='red')
ax.plot([normalized_y_test[:,1].min(), normalized_y_test[:,1].max()], [normalized_y_test[:,1].min(), normalized_y_test[:,1].max()], 'k--', lw=3)
ax.set_xlabel('Target (Murder per Population)')
ax.set_ylabel('Predicted (Murder per Population)')
 
plot_cv_accuracy(cv_accuracy_scores)
plot_cv_rmse(cv_rmse_scores)
plt.show()
