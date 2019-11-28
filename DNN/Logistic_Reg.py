#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt

dataset_path='/home/lucien/Documents/RT-1/PM_DeepLearning/logisticReg/\
data_set_1/cifar-10-batches-py/data_batch_2'

m_train = 1000 
m_test=500		
num_pix=32

classifier=3  #0 avion,1 voiture,2 oiseau,3 chat,4 cerf,5 chien, 6 grenouille
				#7 cheval, 8 bateau, 9 camion
learning_rate=0.0055  
num_iters=2000 

def unpickle(file):
	#ouverture pickle du dictionnaire
	fich=open(dataset_path,'rb')
	dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
	#extraction des images (matrice 3072*10000)
	X=np.array(dict[b'data'].T)
	y=np.asarray(dict[b'labels'])  #shape 10 000,
	return X,y

def initTrainingSet(X,y,m_train,m_test,classifier): 

	nb_of_class1 = int((m_train+m_test)/2)    #modifiable si besoin
	nb_of_class0 = int((m_train+m_test)/2)	 #nb d'elemnts par classe (50/50 ici)
	
	nb_class1_m_train = int(m_train/2)  #nb of class1 in m_train

	idx_1 = np.arange(nb_of_class1)
	idx_0 = np.arange(nb_of_class0)

	index_class1=np.take(np.where(y == classifier),idx_1)
	index_class0=np.take(np.where(y != classifier),idx_0)

	#on met les images correspondant a classe1
	X_train = X[:,index_class1[ 0 : nb_class1_m_train] ]
	X_train = np.concatenate((X_train , X [:,index_class0[0 : nb_class1_m_train ]]), axis=1)

	X_test = X [:,index_class1[ nb_class1_m_train : ] ]
	X_test = np.concatenate((X_test, X[:,index_class0[ nb_class1_m_train :]]),axis=1)

	y_train = np.concatenate(    (    np.ones((nb_class1_m_train,1)), np.zeros((nb_class1_m_train,1)) ) , axis=0 )
	y_test = np.concatenate(      (   np.ones((int(m_test/2),1)) , np.zeros((int(m_test/2),1)) ) , axis=0 )

	X_train = np.concatenate( ( np.ones((1,m_train)), X_train), axis=0)  #rajoute ligne de 1
	X_test =np.concatenate( ( np.ones((1,m_test)), X_test), axis=0)

	print("taille X_train")
	print(np.shape(X_train))
	print("taille X_test")
	print(np.shape(X_test))
	print("taille y_train")
	print(np.shape(y_train))
	print("taille y_test")
	print(np.shape(y_test))

	X_train=X_train.astype(int)
	X_test = X_test.astype(int)

	return X_train,X_test,y_train,y_test

def plotImage(X,idx,num_pix):

	rgb=X[1:,idx]    
	img=rgb.reshape(3,num_pix,num_pix).transpose([1,2,0])
	plt.imshow(img)
	plt.show()

#=============================================================

def initParameters (dim):  #init theta a dim+1 (x0 = 1)
	theta = np.zeros((dim+1,1))
	return theta

def standardizeDataset(X_train,X_test):  #normalisation simplifiée
	
	return (X_train/255),(X_test/255)

def sigmoid(z):
	s= 1 / (1 + np.exp( -z))
	return s

def  costFunction(X,y,theta):
	m=np.shape(y)[0]
	h=sigmoid(np.dot(X.T,theta))
	J = (1/m) * (-   np.dot(y.T,np.log(h)) - np.dot( (1-y).T ,np.log(1-h)) )
	grad = (1/m) * np.dot( X,(h-y))

	return J,grad

def gradDescent(X,y,theta,alpha,num_iters):
	j_history=np.zeros((num_iters,1))

	for iter in range(0,num_iters):
		j_history[iter,0],grad = costFunction(X,y,theta)

		theta = theta - alpha*grad

		if iter %100 ==0 :
			print("iteration %d: cost=%f" %(iter,j_history[iter,0]))

	return j_history,theta

def plotCost(j_history,num_iters,learning_rate):
	x=np.arange(0,num_iters)
	y=j_history
	plt.plot(x,y,linewidth=2.0)
	plt.title("learning_rate =%f" %(learning_rate))
	plt.show()

def predict(X,y,theta):
	s = sigmoid(np.dot(X.T , theta))
	s[np.where(s >= 0.5),0] = 1
	s[np.where(s < 0.5),0 ] = 0
	right_prediction = np.shape(np.where(s==y))[1]
	accuracy = (right_prediction / np.shape(y)[0])*100

	return accuracy

X,y = unpickle(dataset_path)
X_train, X_test, y_train, y_test = initTrainingSet(X,y,m_train,m_test,classifier)
X_train , X_test = standardizeDataset(X_train,X_test)
		
plotImage(X_test,24,num_pix) #chat
plotImage(X_test,25,num_pix) #pas chat

plotImage(X_train,10,num_pix) #chat
plotImage(X_train,30,num_pix)	#pas un chat

theta = initParameters(num_pix*num_pix*3)
j_history,theta = gradDescent(X_train,y_train,theta, learning_rate,num_iters)
plotCost(j_history,num_iters,learning_rate)

accuracy_train=predict(X_train,y_train,theta)
accuracy_test=predict(X_test,y_test,theta)
		
print("alpha=%f, trainSet accuracy :%f testSet accuracy :%f" %(learning_rate,accuracy_train, accuracy_test))