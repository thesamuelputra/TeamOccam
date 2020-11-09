import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

n_hidden1 = 50
n_hidden2 = 25
n_input = 21
n_output = 10

df = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='K:AE, AT')
# Target = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='AT')


val_dataframe = df.sample(frac=0.2, random_state=1337)
train_dataframe = df.drop(val_dataframe.index)

print("Using %d samples for training and %d for validation" % (len(train_dataframe), len(val_dataframe)))

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop('NSP')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

LB = keras.Input(shape=(1,), name="LB")
AC = keras.Input(shape=(1,), name="AC.1")
FM = keras.Input(shape=(1,), name="FM.1")
UC = keras.Input(shape=(1,), name="UC.1")
DL = keras.Input(shape=(1,), name="DL.1")
DS = keras.Input(shape=(1,), name="DS.1")
DP = keras.Input(shape=(1,), name="DP.1")
ASTV = keras.Input(shape=(1,), name="ASTV")
MSTV = keras.Input(shape=(1,), name="MSTV")
ALTV = keras.Input(shape=(1,), name="ALTV")
MLTV = keras.Input(shape=(1,), name="MLTV")
Width = keras.Input(shape=(1,), name="Width")
Min = keras.Input(shape=(1,), name="Min")
Max = keras.Input(shape=(1,), name="Max")
Nmax = keras.Input(shape=(1,), name="Nmax")
Nzeros = keras.Input(shape=(1,), name="Nzeros")
Mode = keras.Input(shape=(1,), name="Mode")
Mean = keras.Input(shape=(1,), name="Mean")
Median = keras.Input(shape=(1,), name="Median")
Variance = keras.Input(shape=(1,), name="Variance")
Tendency = keras.Input(shape=(1,), name="Tendency")

all_inputs = [
    LB,
    AC,
    FM,
    UC,
    DL,
    DS,
    DP,
    ASTV,
    MSTV,
    ALTV,
    MLTV,
    Width,
    Min,
    Max,
    Nmax,
    Nzeros,
    Mode,
    Mean,
    Median,
    Variance,
    Tendency
]
features = layers.concatenate(all_inputs)

x = layers.Dense(n_hidden1, activation="softmax")(features)
x = layers.Dense(n_hidden2, activation="softmax")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# `rankdir='LR'` is to make the graph horizontal.
# keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# print(train_ds.output_shapes)

model.fit(train_ds, epochs=50, steps_per_epoch=10)