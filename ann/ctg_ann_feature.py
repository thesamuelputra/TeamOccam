import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

n_hidden1 = 10
n_hidden2 = 10
n_input = 21
n_output = 10

x = pd.read_excel("C:/Users/david/Downloads/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='K:AE')
y = pd.read_excel("C:/Users/david/Downloads/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='AR')
y = y - 1
# print(x.columns)
# print(y.info())

x_train, x_validate, x_test =  np.split(x.sample(frac=1, random_state=42), [int(0.6*len(x)), int(0.8*len(x))])
y_train, y_validate, y_test =  np.split(y.sample(frac=1, random_state=42), [int(0.6*len(y)), int(0.8*len(y))])

def plot_loss(history):
    plt.figure('loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

def plot_accuracy(history):
    plt.figure('accuracy')
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    # plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

# print(x_train)
# print(y_train)

# x_test = x.sample(frac=0.2, random_state=None)
# y_test = y.sample(frac=0.2, random_state=None)
# x_train = x.drop(x_test.index)
# y_train = y.drop(y_test.index)

# x_train = x.sample(frac=0.7, random_state=0)
# y_train = y.sample(frac=0.7, random_state=0)
# x_test = x.drop(x_train.index)
# y_test = y.drop(y_train.index)

# val_dataframe = x.sample(frac=0.2, random_state=1337)
# train_dataframe = x.drop(val_dataframe.index)

# print("Using %d samples for training and %d for validation" % (len(train_dataframe), len(val_dataframe)))

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = labels.copy()
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 24
train_ds = df_to_dataset(x_train, y_train, batch_size=batch_size)
val_ds = df_to_dataset(x_validate, y_validate, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(x_test, y_test, shuffle=False, batch_size=batch_size)

# feature_columns = []
# for feature_batch in train_ds.take(1):
#   # feature_columns.append(tf.feature_column.numeric_column)
#   print('Every feature:', list(feature_batch.keys()))

feature_columns = []
feature_columns.append(tf.feature_column.numeric_column(key="LB"))
feature_columns.append(tf.feature_column.numeric_column(key="AC.1"))
feature_columns.append(tf.feature_column.numeric_column(key="FM.1"))
feature_columns.append(tf.feature_column.numeric_column(key="UC.1"))
feature_columns.append(tf.feature_column.numeric_column(key="DL.1"))
feature_columns.append(tf.feature_column.numeric_column(key="DS.1"))
feature_columns.append(tf.feature_column.numeric_column(key="DP.1"))
feature_columns.append(tf.feature_column.numeric_column(key="ASTV"))
feature_columns.append(tf.feature_column.numeric_column(key="MSTV"))
feature_columns.append(tf.feature_column.numeric_column(key="ALTV"))
feature_columns.append(tf.feature_column.numeric_column(key="MLTV"))
feature_columns.append(tf.feature_column.numeric_column(key="Width"))
feature_columns.append(tf.feature_column.numeric_column(key="Min"))
feature_columns.append(tf.feature_column.numeric_column(key="Max"))
feature_columns.append(tf.feature_column.numeric_column(key="Nmax"))
feature_columns.append(tf.feature_column.numeric_column(key="Nzeros"))
feature_columns.append(tf.feature_column.numeric_column(key="Mode"))
feature_columns.append(tf.feature_column.numeric_column(key="Mean"))
feature_columns.append(tf.feature_column.numeric_column(key="Median"))
feature_columns.append(tf.feature_column.numeric_column(key="Variance"))
feature_columns.append(tf.feature_column.numeric_column(key="Tendency"))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# model = tf.keras.Sequential([
#   feature_layer,
#   layers.Dense(32, activation='relu'),
#   layers.Dense(32, activation='relu'),
#   layers.Dropout(.1),
#   layers.Dense(3)
# ])

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(n_hidden1, activation='relu'),
  layers.Dense(n_hidden2, activation='sigmoid'),
  layers.Dense(n_output, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=1000)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

plot_loss(history)
plot_accuracy(history)
plt.show()