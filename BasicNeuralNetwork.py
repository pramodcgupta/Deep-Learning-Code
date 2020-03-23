# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:03:06 2020

@author: Pramod.Gupta
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Define dataset
x=[x for x in range(-50,51)]
y=[x**2 for x in x]

# Plot the graph
plt.scatter(x,y)
plt.title('Sqaure Function plotting')
plt.xlabel('X')
plt.ylabel('F(X)')
plt.show()

# Convert to array 
x=np.asarray(x)
y=np.asarray(y)

# Convert to Rows from list
x=x.reshape(len(x),1)
y=y.reshape(len(y),1)

# Normalize the input and output data
from sklearn.preprocessing import MinMaxScaler

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)

# Design Basic Neural Network

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x,y,batch_size=10,epochs=500)

y_pred=model.predict(x)

# Revert back to Original form
x=scale_x.inverse_transform(x)
y=scale_y.inverse_transform(y)
y_pred=scale_y.inverse_transform(y_pred)

# report model error
print('MSE: %.3f' % mean_squared_error(y, y_pred))

# plot x vs y
plt.scatter(x,y, label='Actual')
# plot x vs yhat
plt.scatter(x,y_pred, label='Predicted')
plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()

