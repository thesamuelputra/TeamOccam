import numpy as np
import tensorflow as tf
import pandas as pd

n_hidden1 = 50
n_hidden2 = 25
n_input = 21
n_output = 10

x = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='K:AE')
y = pd.read_excel("datasets/CTG.xls", sheet_name="Data", skiprows=[0, 2128, 2129, 2130], usecols='AR,AT')

print(x.info())
print(y.info())