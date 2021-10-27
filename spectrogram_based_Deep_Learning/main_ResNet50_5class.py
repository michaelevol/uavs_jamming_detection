# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:22:22 2021

@author: spahedin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import SGD
import datetime


trainX = np.load('trainX_5c.npy')
trainY = np.load('trainY_5c.npy')
testX  = np.load('testX_5c.npy')
testY  = np.load('testY_5c.npy')

print ("number of training examples = " + str(trainX.shape[0]))
print ("number of test examples = " + str(testX.shape[0]))
print ("X_train shape: " + str(trainX.shape))
print ("Y_train shape: " + str(trainY.shape))
print ("X_test shape: " + str(testX.shape))
print ("Y_test shape: " + str(testY.shape))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

nb_classes = testY.shape[1]

model = ResNet50(include_top=True, weights= None, input_shape = trainX[0].shape, classes=nb_classes)

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


model.save('.ResNet_5.h5')
