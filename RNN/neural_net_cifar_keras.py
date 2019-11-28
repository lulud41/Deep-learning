#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:43:50 2019

@author: lucien
"""
from keras import models
from keras import layers
from keras import optimizers


import numpy as np     #faire plusieurs focntions : descente grad, lambda et dropout
import pickle
import matplotlib.pyplot as plt

dataset_path='/home/lucien/Documents/RT-1/PM_DeepLearning/logisticReg/data_set_1/cifar-10-batches-py/'

def initDatasets(m_train,m_test):
    
	#ouverture pickle du dictionnaire
    fich=open(dataset_path+"data_batch_"+str(1),'rb')
    dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
	#extraction des images (matrice 3072*10000)
    X_train=np.array(dict[b'data'].T)
    y_train=np.asarray(dict[b'labels'])  #shape 10 000,
    
    for i in range(2,6):
        fich=open(dataset_path+"data_batch_"+str(i),'rb')
        dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
   # 	#extraction des images (matrice 3072*10000)
        X_train=np.concatenate((X_train , np.asarray(dict[b'data'].T) ),axis = 1)
        y_train=np.concatenate((y_train, np.asarray(dict[b'labels'])))
 
    fich=open(dataset_path+"test_batch",'rb')
    dict = pickle.load(fich,encoding='bytes')
    X_test= np.asarray(dict[b'data'].T)
    y_test= np.asarray(dict[b'labels'])
    
    X_train = X_train[:,0:m_train]
    y_train = y_train[0:m_train]
    X_test = X_test[:,0:m_test]
    y_test = y_test[0:m_test]
    
    #mise des Y sous forme vectorielle : passe de y=4 pour classe 5 à y=[0 0 0 0 1 0 0 0 0]
    y_test_M = np.zeros((m_test,10))
    y_train_M = np.zeros((m_train,10))
    
    for i in range(0,m_test):
        y_test_M[i,y_test[i]] = 1
    for i in range(0,m_train):
        y_train_M[i,y_train[i]] = 1
    
    return X_train,y_train_M,X_test,y_test_M

X_train, y_train, X_test, y_test = initDatasets(10000,300)
X_train=(X_train-np.mean(X_train) )/np.std(X_train)
X_test=(X_test-np.mean(X_test))/np.std(X_test)

model = models.Sequential()
model.add(layers.Dense(1024, activation='relu',input_shape=(3072,)))
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train.T,y_train,epochs=5,batch_size=64,validation_data=(X_test.T,y_test))

train_loss = history.history['loss']
test_loss = history.history['val_loss']
acc_train = history.history['acc']
acc_test = history.history['val_acc']

print("performance : train accuracy :",acc_train[len(acc_train)-1]," test accuracy",acc_test[len(acc_test)-1])

x=range(0,len(test_loss))
plt.plot(x,history.history['loss'],'b',label='Training loss')
plt.plot(x,history.history['val_loss'],'r',label='Test loss')
plt.title('training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()



