#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 18:27:29 2019

@author: lucien
"""


# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/



import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
import keras

from keras.preprocessing.image import ImageDataGenerator



data_path="/home/lucien/Documents/keras_studies/dog_vs_cat_data/data_subset/"

def init_model():
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3),activation='relu', input_shape=(50,50,3)))
    model.add(layers.MaxPooling2D(2,2))
    
    model.add(layers.Conv2D(64, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model

    
def data_preprocessing():
    train_data_generator = ImageDataGenerator(rescale=1./255,
                                              rotation_range=40,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)
    
    train_generator = train_data_generator.flow_from_directory(data_path+"train/",
                                                               target_size=(50,50),
                                                               batch_size=20,
                                                               class_mode="binary")
    
    validation_data_generator = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_data_generator.flow_from_directory(data_path+"validation/",
                                                                         target_size=(50,50),
                                                                         batch_size=20,
                                                                         class_mode="binary")
    
    return train_generator, validation_generator

def plot_cost(history):
    epochs = range(0,len(history.history['loss']))
    
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss =history.history['val_loss']
    val_acc = history.history['val_acc']
    
    plt.plot(epochs,loss,'xb',label = "training loss")
    plt.plot(epochs,val_loss,'r',label= "validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss graph")
    plt.legend()
    plt.figure()
    
    plt.plot(epochs,acc,"xb",label = "training accuracy")
    plt.plot(epochs,val_acc,"r", label ="validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Accuracy graph")
    plt.legend()
    
    print("accuracy on trainSet : ",acc[len(acc)-1]," accuracy on testSet : ", val_acc[len(val_acc)-1] )
    
    plt.show()
 

def train_model():
        
    call_backs_list = [
        keras.callbacks.EarlyStopping(
            monitor='acc', patience = 1),
        keras.callbacks.ModelCheckpoint(
            filepath = 'dog_vs_cat_model_check_point.h5',
            monitor = 'val_loss',
            save_best_only = True,)
    ]


    model = init_model()
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

    train_generator, validation_generator = data_preprocessing()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks = call_backs_list
        )

    plot_cost(history)
    
    model.save('cats_vs_dogs_1_h5')


train_model()


    
    
    
    
    
    
    
    