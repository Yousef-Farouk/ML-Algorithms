#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras import optimizers


# In[18]:


(x_train,y_train_1),(x_test,y_test_1)=datasets.mnist.load_data()


# In[14]:


y_train=tf.keras.utils.to_categorical(y_train_1,10)
y_test=tf.keras.utils.to_categorical(y_test_1,10)          # y_train_1 = 5  ==> y_train==[0000010000]


# In[15]:


print(x_train.shape)


# In[16]:


x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)


# In[17]:


print(x_train.shape)


# In[6]:


#Normalization
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# In[7]:


#ANN

model=Sequential([Dense(512,input_dim=784,activation='relu'),Dense(256,activation='relu'),Dense(124,activation='relu'),Dense(10,activation='softmax')])
model.summary()


# In[8]:


#optimization

from tensorflow.keras import optimizers
opt =optimizers.Adam(0.001)     #learning rate


# In[9]:


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=10)


# In[10]:


test_loss,test_accuracy=model.evaluate(x=x_test,y=y_test)
print("the test accuracy is: " ,test_accuracy)


# In[ ]:




