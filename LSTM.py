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

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
X_train , y_train = np.array (X_train), np.array(y_train)

# Reshaping 
X_train = X_train.reshape ( X_train, (X_train.shape[0], X_train.shape[1], 1))