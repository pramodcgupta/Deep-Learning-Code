# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:00:54 2020

@author: Pramod.Gupta
"""

# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

X=dataset[:,0:8]
y=dataset[:,8]

# Step 1: Create ANN

model=Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(150, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X,y,epochs=1000, batch_size=100)

# Evaluate Performance
_, accuracy=model.evaluate(X,y)

print("Accuarcy: ", accuracy)

# Predict for new data
y_pred=model.predict_classes(X)

for i in range(5):     
    print(X[i].tolist(),":", y_pred[i], " : ", y[i])

