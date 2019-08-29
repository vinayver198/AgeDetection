from keras.models import model_from_json
from keras.utils import to_categorical
import os
import random as rn
import pandas as pd
import  matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle


test_filePath = "./test_data/Test/"
test  = pd.read_csv('test_data/test.csv')


temp = []
for img_name in test['ID'] :
       image = plt.imread(test_filePath + img_name)
       image = cv2.resize(image,(32,32))
       temp.append(image)       
       
test_x = np.stack(temp)
test_x = test_x/255

json_file = open('model.json')
architecture = json_file.read()
json_file.close()

model = model_from_json(architecture)
model.load_weights('model_weights.h5')
lE = pickle.load('labelEncoder.pickle')


y_pred_rgb = model.predict(test_x)
y_pred_rgb = np.argmax(y_pred_rgb,axis=1)
y_pred_rgb = list(lE.inverse_transform(y_pred_rgb))
plt.imshow(test_x[206])
plt.title(y_pred_rgb[206])
