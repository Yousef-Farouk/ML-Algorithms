# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 05:37:18 2022

@author: hesham
"""
###
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
from keras.optimizers import adadelta_v2
from keras.utils import np_utils


epochs=10
learning_rate= 0.001
batch_size=128
num_classes =10

### to clarify input images dimenssions 

img_rows ,img_cols = 28,28
#To load the data  and spliting it into variables use:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#To print the shape of the training and testing vectors use :
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))

### ### and to display it :
#img2=train_X[num of index of image]
#plt.imshow(img2)

img2=X_train[9]
plt.imshow(img2)
######################################
### then we will reshape the images 

X_train =X_train.reshape(60000, 28, 28,1)

X_test=X_test.reshape(10000, 28, 28,1)

#### then we will turn the class vector tobinary class matrises
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)

y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)

####################################################################
## then we will make the cnn

model=Sequential()
### then we will use 32 filters with a size of 3*3 
### and we will use relu activation func to add the nonlinearity 

model.add(Conv2D(32,3,3, padding='same',input_shape=(28,28,1), activation='relu'))

## now we had done conv layer
### and we will make another conv layer  
model.add(Conv2D(64,(3,3), activation='relu'))
#### then we will make a max pooling layer 

model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
### 

### then we will use droupout to delete some cells to reduse the overfitting 
   
model.add(Dropout(0.2))
 
## then we will use the flatten layer 
model.add(Flatten())
##
## then we will make the fully connected layer 
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

### at the last dense the first num or the output must equal the num of classes and we use softmax to make classification 
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

############# now we had built aneural network bu we hadnt trained it it 
###############################################################################

### now we must compile our model  to add more informations before learning 
## and metrics is the value which appear while making trainning 
model.compile (loss=keras.losses.CategoricalCrossentropy(),
               metrics=['accuracy'])

## then we will make training

model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs, batch_size=batch_size, verbose=2)

## and then to calculate the score and print test loss and test accuracy 

score= model.evaluate(X_test, y_test, verbose=0)
print ('test loss ',score[0])
print ('test accuaracy ',score[1])




















































