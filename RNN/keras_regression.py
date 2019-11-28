#!/usr/bin/python3
# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras import optimizers


import numpy as np     #faire plusieurs focntions : descente grad, lambda et dropout
import matplotlib.pyplot as plt



from keras.datasets import boston_housing


(train_data, train_targets) , (test_data, test_target) = boston_housing.load_data()

train_data= (train_data - np.mean(train_data,axis=0) ) / np.std(train_data,axis=0)
test_data = (test_data - np.mean(test_data,axis=0) ) / np.std(test_data,axis=0)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

model = build_model()

history=model.fit( train_data,train_targets,batch_size=1,epochs=20,
          validation_data=(test_data,test_target))

acc_train = history.history["val_mean_absolute_error"]
acc_test = history.history["mean_absolute_error"]
print("perf acc train" ,acc_train[len(acc_train)-1]," acc test :",acc_test[len(acc_test)-1])

loss_train = history.history["loss"]
loss_test = history.history["val_loss"]
x= np.arange(0,len(loss_test))
plt.plot(x,loss_train,label='train loss')
plt.plot(x,loss_test,label='test loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()



