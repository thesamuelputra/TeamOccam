import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


n_hidden1 = 10
n_hidden2 = 10
n_input = 21
n_output = 10

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

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


def plot_cv_loss(average_loss):
    plt.figure('cv_loss')
    plt.plot(average_loss, label='loss')
    plt.plot(average_loss, 'ro')
    plt.ylim([0, 1])
    plt.xlabel('folds')
    plt.ylabel('loss')
    plt.legend()

x = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='K:AE')
y = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='AR')
y = y - 1
# y = to_categorical(y, n_output)

x = normalize(x)
y = normalize(y)

batch_x_train = x.sample(frac=0.8, random_state=0)
batch_y_train = y.sample(frac=0.8, random_state=0)
batch_x_test = x.drop(batch_x_train.index).values
batch_y_test = y.drop(batch_y_train.index).values
batch_x_train = batch_x_train.values
batch_y_train = batch_y_train.values

# prepare cross validation
kf = KFold(n_splits=10, random_state=5, shuffle=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(n_input,), activation='tanh'),
    tf.keras.layers.Dense(n_hidden2, activation='relu'),
    tf.keras.layers.Dense(n_output, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

k_fold = 1
cv_accuracy_scores = []
cv_rmse_scores = []
cv_loss_scores = []

for train_index, test_index in kf.split(batch_x_train):
    print("Cross fold validation - fold {}".format(k_fold))
    x_train, x_test = batch_x_train[train_index], batch_x_train[test_index]
    y_train, y_test = batch_y_train[train_index], batch_y_train[test_index]
    history = model.fit(x_train, y_train, epochs=125, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    cv_accuracy_scores.append(accuracy)
    # cv_rmse_scores.append(rmse)
    cv_loss_scores.append(loss)
    k_fold += 1


print("-" * 70)
print("Average accuracy: {}".format(np.mean(cv_accuracy_scores)))
#print("Average root mean square error: {}".format(np.mean(cv_rmse_scores)))
#loss, accuracy, rmse = model.evaluate(batch_x_test, batch_y_test)

loss, accuracy = model.evaluate(batch_x_test, batch_y_test)
print("Test result: loss ({}), accuracy ({})".format(loss, accuracy))

# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


plot_cv_accuracy(cv_accuracy_scores)
plot_cv_loss(cv_loss_scores)
plt.show()