#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:53:02 2018

@author: lucien
"""
from tqdm import tqdm  # 1000*1000 calcul raisonnable    A changer; normalisation+init des inputs + descente grad
import numpy as np     #faire plusieurs focntions : descente grad, lambda et dropout
import pickle
import matplotlib.pyplot as plt

dataset_path='/home/lucien/Documents/RT-1/PM_DeepLearning/logisticReg\
/data_set_1/cifar-10-batches-py/data_batch_2'

m_train =400 #nb pair, moitié y=classifier moitié autre
m_test=200		#pareil

num_pix=32
n_inputLayer=num_pix*num_pix*3   #number of elements in input layer

networkArch=[n_inputLayer,1000,1000,1]  #number of elements of each layer   
#79% avec input,20,1 ,LR=0.05,2500,lam=0.5

classifier=0      #0 avion,1 voiture,2 oiseau,3 chat,4 cerf,5 chien, 6 grenouille
				#7 cheval, 8 bateau, 9 camion
learning_rate=0.05
num_iters=1000
lambd=5 #80% avec mtrain=800,20 n_h,lR =0.08,numIter=2500,lambd=4

def unpickle(file):
	#ouverture pickle du dictionnaire
	fich=open(dataset_path,'rb')
	dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
	#extraction des images (matrice 3072*10000)
	X=np.array(dict[b'data'].T)
	y=np.asarray(dict[b'labels'])  #shape 10 000,
	return X,y

def initTrainingSet(X,y,m_train,m_test,classifier):   
	#dataset_size = np.shape(np.where(y==classifier))[1]
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

    y_train = np.concatenate(( np.ones((nb_class1_m_train,1)),np.zeros((nb_class1_m_train,1))),axis=0)
    y_test = np.concatenate( (   np.ones((int(m_test/2),1)) , np.zeros((int(m_test/2),1)) ),axis=0 )

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

	rgb=X[:,idx]    
	img=rgb.reshape(3,num_pix,num_pix).transpose([1,2,0])
	plt.imshow(img)
	plt.show()

def standardizeDataset(X_train,X_test):  #normalisation simplifiée	
    return (X_train/255),(X_test/255)

def initialiseParameters(networkArch):
    parameters = dict()
    L = len(networkArch)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(networkArch[l],networkArch[l-1])*0.01
        parameters["b"+str(l)] = np.zeros((networkArch[l],1))    
    return parameters

def ReLu(z):
    z[np.where(z <= 0)] = 0
    return z

def sigmoid(z):
	s= 1 / (1 + np.exp( -z))
	return s

def derivativeRelu(z):
    z[np.where(z <= 0)] = 0
    z[np.where(z > 0)] = 1
    return z

def derivativeSigmoid(z):
    z = sigmoid(z)*(1 - sigmoid(z) )
    return z
    
def forwardPropagation(parameters,X):
    
    cache=dict()
    L = len(parameters)//2 #np de layers
    cache["A"+str(0)] = X
    A=X
    for l in range(1,L):
        Z = np.dot(parameters["W"+str(l)],A) + parameters["b"+str(l)]
        cache["Z"+str(l)] = Z
        A=ReLu(Z)
        cache["A"+str(l)] = A
        
    Z_L = np.dot(parameters["W"+str(L)] , A) + parameters["b"+str(L)]
    cache["Z"+str(L)] = Z_L
    A_L = sigmoid(Z_L)
    cache["A"+str(L)] = Z_L
    
    return A_L , cache

def computeCost(A_L , y_train, parameters,lambd ):
    L = len(parameters)//2
    regularisation = 0
    m=np.shape(y_train)[0]

    for l in range(1,L+1):
        regularisation = regularisation + np.sum( np.square(parameters["W"+str(l)]) )

    J = (1/m) * ( -np.dot( np.log(A_L) ,y_train) - np.dot( np.log(1-A_L),(1-y_train) ) )
    J = J + regularisation*lambd/(2*m)
    return J

def derivativeCostFunction(A_L,y):
    dAL=- (np.divide(y.T, A_L) - np.divide(1 - y.T, 1 - A_L)) 
    return dAL

def backPropagation(parameters, y_train, A_L, cache,lambd):
    L = len(parameters)//2
    m = np.shape(y_train)[0]
    grads=dict()

    dAL = derivativeCostFunction(A_L ,y_train)
    dZL = dAL*derivativeSigmoid(cache["Z"+str(L)])
  
    grads["dW"+str(L)] = (1/m)* np.dot( dZL , cache["A"+str(L-1)].T )+(lambd/m)*parameters["W"+str(L)]
    grads["db"+str(L)] = (1/m)* np.sum( dZL, axis=1, keepdims=True)
    dAl_1 = np.dot( parameters["W"+str(L)].T , dZL)
    
    for l in reversed(range(1,L)):
        
        dZl = dAl_1 * derivativeRelu(cache["Z"+str(l)])
        grads["dW"+str(l)] = (1/m)* np.dot( dZl , cache["A"+str(l-1)].T ) +(lambd/m)*parameters["W"+str(l)]
        grads["db"+str(l)] = (1/m)* np.sum( dZl, axis=1, keepdims=True)
        dAl_1 = np.dot( parameters["W"+str(l)].T , dZl )

    return grads

def updateParameters(parameters,grads,Learning_rate):
    L = len(parameters)//2

    for l in range(1,L+1):
        
        parameters["W"+str(l)] = parameters["W"+str(l)] - Learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - Learning_rate*grads["db"+str(l)]
    return parameters

def plotCost(j_history,num_iters,learning_rate):
    x=np.arange(0,num_iters)
    y=j_history
    plt.plot(x,y,linewidth=2.0)
    plt.title("learning_rate =%f" %(learning_rate))
    plt.show()

def predict(X,y,parameters):
    A_L,cache = forwardPropagation(parameters,X) #A4 : 1,200 et y: 200,1
    A_L=A_L.T
    A_L[np.where(A_L >= 0.5)] = 1
    A_L[np.where(A_L < 0.5)] = 0
    right_prediction = np.shape(np.where(A_L==y))[1]
    accuracy = (right_prediction / np.shape(y)[0])*100

    return accuracy

X,y = unpickle(dataset_path)
X_train, X_test, y_train, y_test = initTrainingSet(X,y,m_train,m_test,classifier)
X_train , X_test = standardizeDataset(X_train,X_test)

plotImage(X_train,0,32)
plotImage(X_test,100,32)

parameters = initialiseParameters(networkArch)
j_history=np.zeros((num_iters,1))
J=0

pbar = tqdm(range(1,num_iters))

for iter in pbar:
    
    A_L , cache = forwardPropagation(parameters,X_train)
    J = computeCost(A_L,y_train,parameters,lambd)
    
    j_history[iter,0] = J

    grads = backPropagation(parameters,y_train,A_L,cache,lambd) 
    
    parameters=updateParameters(parameters,grads,learning_rate)
    if iter %10 ==0 :
        pbar.set_description("cost : %f" %J)
    
    
    accuracy_train=predict(X_train,y_train,parameters)
    accuracy_test=predict(X_test,y_test,parameters)

print("alpha=%f, trainSet:%f testSet:%f" %(learning_rate,accuracy_train, accuracy_test))
plotCost(j_history,num_iters,learning_rate)