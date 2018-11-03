import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: , 1:2].values

sc = MinMaxScaler( feature_range= (0,1))

training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 time steps
X_train = []
y_train = []

for i in range(60, 1258)