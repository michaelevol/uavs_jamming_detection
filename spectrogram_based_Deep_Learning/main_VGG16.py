# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:01:28 2021

@author: spahedin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import SGD
from keras.models import Model
import datetime

trainX = np.load('trainX_5c.npy')
trainY = np.load('trainY_5c.npy')
testX  = np.load('testX_5c.npy')
testY  = np.load('testY_5c.npy')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

nb_classes = 5


def VGG16(input_shape):  # VGG - D configurations
    X_input = Input(input_shape)
    
    # Block 1
    X = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(X_input)
    X = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv2')(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(X)
    
    # Block 2
    X = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv2')(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(X)
    
    # Block 3
    X = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1')(X)
    X = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2')(X)
    X = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3')(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(X)
    
    # Block 4
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv1')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv3')(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(X)
    
    # Block 5
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv1')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv3')(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(X)
    
    X = Flatten()(X)
    
    X = Dense(4096, activation = 'relu', name = "fc0")(X)
    
    X = Dense(4096, activation = 'relu', name = 'fc1')(X) 
    
    
    X = Dense(nb_classes,activation='softmax',name = 'fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='vgg16')
    
    return model

model = VGG16(trainX[0].shape)
model.summary()
opt = SGD(learning_rate=0.0001)
model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=['accuracy'])

theTime = datetime.datetime.now()
print(theTime)
history = model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=1)
theTime2 = datetime.datetime.now()
print(theTime2)
print(theTime2-theTime)

theTime3 = datetime.datetime.now()
print(theTime3)
y_pred_keras = model.predict(testX)
theTime4 = datetime.datetime.now()
print(theTime4)
print(theTime4-theTime3)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr_keras = dict()
tpr_keras = dict()
auc_keras= dict()
for i in range(nb_classes):
    fpr_keras[i],tpr_keras[i], _ = roc_curve(testY[:,i],y_pred_keras[:,i])
    auc_keras[i] = auc(fpr_keras[i],tpr_keras[i])


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
colors = ['red','blue','yellow','green','navy']
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr_keras[i], tpr_keras[i], color = color, label='Keras (area = {:.3f})'.format(auc_keras[i]))
plt.legend(loc='lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Acc'], loc = 'upper left')
plt.show()

plt.figure(3)
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss'], loc = 'upper left')
plt.show()

score = model.evaluate(testX, testY, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

from sklearn.metrics import confusion_matrix
predictions = model.predict(testX)
print(confusion_matrix(np.argmax(testY, axis=1), np.argmax(y_pred_keras, axis=1)))


model.save('.VGG16_5.h5')
