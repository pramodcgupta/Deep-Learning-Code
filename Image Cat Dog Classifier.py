# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:17:44 2020

@author: Pramod.Gupta
"""

# Building CNN

# Importing all libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initialize model
classifier = Sequential()

# 1: Convolution 
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation="relu")) 

# 2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second layer
classifier.add(Conv2D(32, (3,3), activation="relu")) 
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatten dataset
classifier.add(Flatten())

# Adding Dense layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

## Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('./PetImages',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('./PetImages',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,    
                         validation_steps = 2000)


# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('./PetImages/Cat/Img001.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)
