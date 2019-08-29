# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:25:40 2019

@author: vinayver
"""
import os
import random as rn
import pandas as pd
import  matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
filepath = "./train_DETg9GD/Train/"
test_filePath = "./test_data/Test/"


train = pd.read_csv('train_DETg9GD/train.csv')
test  = pd.read_csv('test_data/test.csv')

filename = rn.choice(train['ID'])
image = plt.imread(filepath+filename)
plt.imshow(image)



# Resizing the images
temp = []
for img_name in train['ID'] :
       image = plt.imread(filepath + img_name)
       image = cv2.resize(image,(32,32))
       temp.append(image)       
       
train_x = np.stack(temp)

temp = []
for img_name in test['ID'] :
       image = plt.imread(test_filePath + img_name)
       image = cv2.resize(image,(32,32))
       temp.append(image)       
       
test_x = np.stack(temp)


# Lets normalize our images 
train_x = train_x/255
test_x = test_x/255

# Let's transform our categorical variable
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

lE = LabelEncoder()
train_y = lE.fit_transform(train['Class'])
pickle.dump(lE,open("labelEncoder.pickle","wb"))
train_y = to_categorical(train_y)

input_dim = (32,32,3)
hidden_units = 500
ouput_num_units = 3

epochs = 50
batch_size = 128

from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(filters = 64,kernel_size=(3,3),input_shape = (32,32,3),activation = 'relu',padding = 'same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 128,kernel_size=(3,3),activation = 'relu',padding = 'same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(2,2))

model.add(Flatten())


model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=3,activation='softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(train_x,train_y,batch_size = 128,epochs=epochs,validation_split=0.2, callbacks=callbacks_list)



model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)

y_pred_rgb = model.predict(test_x)
y_pred_rgb = np.argmax(y_pred_rgb,axis=1)
y_pred_rgb = list(lE.inverse_transform(y_pred_rgb))
plt.imshow(test_x[206])
plt.title(y_pred_rgb[206])


#plt.figure(figsize=(20,10))
#plt.subplot(1, 2, 1)
#plt.suptitle('Optimizer : Adam', fontsize=10)
#plt.ylabel('Loss', fontsize=16)
#plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.legend(loc='upper right')
#
#plt.subplot(1, 2, 2)
#plt.ylabel('Accuracy', fontsize=16)
#plt.plot(history.history['acc'], label='Training Accuracy')
#plt.plot(history.history['val_acc'], label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.show()


