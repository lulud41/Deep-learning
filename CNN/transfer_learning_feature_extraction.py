#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:30:38 2019

transfer learning from VGG16, first compute ativation from CNN, then train new Dense model
from scratch

-> no data augmentation



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

batch_size=20
target_size=(50,50)   # si changée : refaire tous les calculs : supprimer les fichiers .npy

conv_model = VGG16(weights ="imagenet",include_top=False, input_shape=(target_size[0],target_size[1],3 ))


def extract_features(nb_samples, directory):
    i=0
    print("processing batches from ",directory)
    bottleneck_features = np.zeros((nb_samples,4,4,512))
    labels = np.zeros(nb_samples,)
    data_generator = ImageDataGenerator(rescale=1./255)
    
    generator = data_generator.flow_from_directory(directory,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode="binary")

    for input_batch,labels_batch in generator:
        bottleneck_features[ i*batch_size : (i+1)*batch_size , :,:,:] = conv_model.predict(input_batch)
        labels[ i*batch_size : (i+1)*batch_size ] = labels_batch
        i=i+1       
       
        if i%10 == 0:
           print("processing batch n° ",i) 
           
        if i*batch_size >= nb_samples:
            print("fin")
            break
    
    return bottleneck_features,labels
    

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
 
def init_model():
    model = models.Sequential()
    model.add(layers.Dense(256,activation='relu', input_dim = 4*4*512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  metrics=["acc"],
                  optimizer=optimizers.RMSprop(lr=2e-5))
    
    return model

def train_model():
    
    model = init_model()
    
    if not os.path.isfile('bottleneck_features_train.npy'):
        train_extracted_features, labels_train = extract_features(nb_train_images, train_path)
        np.save('bottleneck_features_train_labels.npy',labels_train)
        np.save('bottleneck_features_train.npy', train_extracted_features)

    if not os.path.isfile('bottleneck_features_test.npy'):
        test_extracted_features, labels_test = extract_features(nb_test_images, test_path)
        np.save('bottleneck_features_test_labels.npy',labels_test)
        np.save('bottleneck_features_test.npy',test_extracted_features)
        
    train_features = np.load('bottleneck_features_train.npy')
    train_features = np.reshape(train_features, (2000,4*4*512))
    
    test_features = np.load('bottleneck_features_test.npy')
    test_features = np.reshape(test_features,(1000,4*4*512))
    
    labels_train = np.load('bottleneck_features_train_labels.npy')
    labels_test =  np.load('bottleneck_features_test_labels.npy')
    
    history=model.fit(train_features, labels_train,epochs=15,batch_size=20,
              validation_data = (test_features, labels_test))
    
    plot(history)

train_model()
