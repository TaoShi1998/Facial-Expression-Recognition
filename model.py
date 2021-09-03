#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:25:09 2020

@author: Tao Shi
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from preprocessing import width, height
from tensorflow.keras.utils import plot_model
import pydot
 
NUM_FEATURES = 64 
NUM_LABELS = 7
WIDTH, HEIGHT = width, height

# 搭建模型 
def createModel():
    model = Sequential() # linear stack of layers(put layers on top of each other)
    
    # 2-dimensional Convolutional Layer: performs the convolution operation
    model.add(Conv2D(NUM_FEATURES, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    # 'BatchNormalization'performs the batch normalization operation on inputs to the next layer
    # so that the inputs are scaled to 0 to 1
    model.add(BatchNormalization())
    # Pooling operation takes a pooling window of 2 * 2 and a stride of 2 * 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Dropout is a technique where randomly selected neurons are ignored during the training
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2 * 2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2 * 2 * 2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * NUM_FEATURES, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
     
    # flatten the input from N-Dimension to 1-Dimension 
    model.add(Flatten())
     
    model.add(Dense(2 * 2 * 2 * NUM_FEATURES, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * NUM_FEATURES, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * NUM_FEATURES, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(NUM_LABELS, activation='softmax'))
    
    # Compile the model with adam optimizer and categorical crossentropy loss
    model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss = categorical_crossentropy,
                  metrics = ['accuracy'])
    
    return model
 

model = createModel()

if __name__ == '__main__':
    model.summary()
    plot_model(model, to_file = 'test.png', show_shapes = True, show_layer_names = False)



