#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 09:33:09 2019


Step 1 : train classifier on top of freezed CNN
Step 2 : fine tune (train again) classifier AND unfreezed last layers of CNN





@author: lucien
"""


from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16

import numpy as np
import matplotlib.pyplot as plt

import os

data_path = "/home/lucien/Documents/keras_studies/dog_vs_cat_data/data_subset/"
train_path = os.path.join(data_path,"train/")
test_path = os.path.join(data_path,"test/")

nb_train_images = len(os.listdir(train_path+"/train_cats/"))+len(os.listdir(train_path+"/train_dogs/")) 

nb_test_images = len(os.listdir(test_path+"/test_cat/")) + len(os.listdir(test_path+"/test_dog/"))

batch_size=8
target_size=(32,32)   # si changée : refaire tous les calculs : supprimer les fichiers .npy



def init_model(step): 
    
    conv_model = VGG16(weights ="imagenet",include_top=False, input_shape=(target_size[0],target_size[1],3 ))  
 
    model = models.Sequential()
    model.add(conv_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    
    if step=="step2": 
        model.load_weights("transfer_learning_fine_tuning_weights.h5")
     
        for layer in model.get_layer("vgg16").layers[ :-4]:
          layer.trainable = False
        
        print("\nmodel architecture \n")
        
        for layer in model.get_layer("vgg16").layers:
          print(layer.name," trainable : ", layer.trainable)
          
    model.get_layer("vgg16").summary()
    
    if step == "step1":
        model.get_layer("vgg16").trainable = False
        
    return model

def data_preprocessing():
    
    train_data_generator = ImageDataGenerator(rescale=1./255,
                                          rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)
     
    train_generator = train_data_generator.flow_from_directory(train_path,
                                                           target_size=target_size,
                                                           batch_size=batch_size,
                                                           class_mode="binary")
    
    test_data_generator = ImageDataGenerator(rescale=1./255)
    test_generator = test_data_generator.flow_from_directory(test_path,
                                                                     target_size=target_size,
                                                                     batch_size=batch_size,
                                                                     class_mode="binary")
    
    return train_generator, test_generator

def plot(history):

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



print("Step 1 : train classifier on top of freezed CNN")
print("Step 2 : fine tune (train again) classifier AND unfreezed last layers of CNN")



if not os.path.isfile("transfer_learning_fine_tuning_weights.h5"):

    model = init_model("step1")
    train_generator , test_generator = data_preprocessing()
        
    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                  loss="binary_crossentropy",
                  metrics=["acc"])
    history = model.fit_generator(train_generator,steps_per_epoch=50,epochs=10,
                        validation_data=test_generator,validation_steps=25)
    
    plot(history)
    
    model.save_weights("transfer_learning_fine_tuning_weights.h5")
    
else:
    print("Step1 already done, loading model")
    
    model = init_model("step2")
    print(model.summary())
    print("begining fine tuning : step 2")
   
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss="binary_crossentropy",
                  metrics=["acc"])
    
    train_generator , test_generator = data_preprocessing()

    history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,
                        validation_data=test_generator,validation_steps=25)
    
    plot(history)   




