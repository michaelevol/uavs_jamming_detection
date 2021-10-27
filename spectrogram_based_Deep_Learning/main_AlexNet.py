# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:01:28 2021

@author: spahedin
"""

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from keras.models import Model
import datetime

trainX = np.load(r'F:\Yuchen\Thesis4\class5\trainX_5c.npy')
trainY = np.load(r'F:\Yuchen\Thesis4\class5\trainY_5c.npy')
testX  = np.load(r'F:\Yuchen\Thesis4\class5\testX_5c.npy')
testY  = np.load(r'F:\Yuchen\Thesis4\class5\testY_5c.npy')


nb_classes = 5

def AlexNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPool2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPool2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPool2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(4096, activation = 'relu', name = "fc0")(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation = 'relu', name = 'fc1')(X) 
    X = Dropout(0.5)(X)
    X = Dense(nb_classes,activation='softmax',name = 'fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AlexNet')
    
    return model

alexNet = AlexNet(trainX[0].shape)
alexNet.summary()

opt = SGD(learning_rate=0.001)

alexNet.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=['accuracy'])

theTime = datetime.datetime.now()
print(theTime)
history = alexNet.fit(trainX,trainY,epochs=5,batch_size=64, verbose=1)
theTime2 = datetime.datetime.now()
print(theTime2)
print(theTime2-theTime)

theTime3 = datetime.datetime.now()
print(theTime3)
y_pred_keras = alexNet.predict(testX)
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

score = alexNet.evaluate(testX, testY, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

from sklearn.metrics import confusion_matrix
predictions = alexNet.predict(testX)
print(confusion_matrix(np.argmax(testY, axis=1), np.argmax(y_pred_keras, axis=1)))


alexNet.save('.AlexNet_5.h5')
