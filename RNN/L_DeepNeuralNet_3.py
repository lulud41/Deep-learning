#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:53:02 2018

@author: lucien
"""
from tqdm import tqdm  # 1000*1000 calcul raisonnable    A changer;
# normalisation+init des inputs + descente grad
import numpy as np     #faire plusieurs focntions : descente grad, lambda et dropout
import pickle
import matplotlib.pyplot as plt

dataset_path='/home/lucien/Documents/RT-1/PM_DeepLearning/logisticReg/data_set_1/cifar-10-batches-py/'
m_train =700# max 49 999
m_test=300	#max 9 999
mini_batch_size =8
num_pix=32
n_inputLayer=num_pix*num_pix*3   #number of elements in input layer

networkArch=[n_inputLayer,50,10]  #number of elements of each layer   

keepProbs = [1,0.9,1]  # dropOut regularisation, one value for each layer (prob of deleting a neuron)
# Batchwise dropout : same mask for whole mini-batch
learning_rate=0.0005  #mettre learning rate basse, et pas de bias correction
num_iters=30
#adam optimizer parameters
epsilon = 1e-8
beta1 =  0.9
beta2 =  0.999

def initDatasets(m_train,m_test):
    
	#ouverture pickle du dictionnaire
    fich=open(dataset_path+"data_batch_"+str(1),'rb')
    dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
	#extraction des images (matrice 3072*10000)
    X_train=np.array(dict[b'data'].T)
    y_train=np.asarray(dict[b'labels'])  #shape 10 000,
    

    # Charge tous le dataset : mais trop lourd pour ma petite RAM

    #=========================================
   # for i in range(2,6):
   #     fich=open(dataset_path+"data_batch_"+str(i),'rb')
   #     dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
   # 	#extraction des images (matrice 3072*10000)
   #     X_train=np.concatenate((X_train , np.asarray(dict[b'data'].T) ),axis = 1)
   #     y_train=np.concatenate((y_train, np.asarray(dict[b'labels'])))
   # Charge tous les autres batch bouffe de la ram
    #======================================
    
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

def plotImage(X,idx,num_pix):

	rgb=X[:,idx]    
	img=rgb.reshape(3,num_pix,num_pix).transpose([1,2,0])
	plt.imshow(img)
	plt.show()

def standardizeDataset(X_train,X_test):
    #normalisation simplifiée	
    return (X_train - np.mean(X_train))/np.std(X_train) , (X_test - np.mean(X_test))/np.std(X_test)

def initialiseParameters(networkArch):

    parameters = dict()
    L = len(networkArch)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(networkArch[l],networkArch[l-1])*np.sqrt(2/networkArch[l-1])
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
    
def dropOutforwardPropagation(parameters,X,keepProbs):
    
    cache=dict()
    L = len(parameters)//2 #np de layers
    cache["A"+str(0)] = X
    A=X
   
    for l in range(1,L):
        Z = np.dot(parameters["W"+str(l)],A) + parameters["b"+str(l)]
        cache["Z"+str(l)] = Z
        A=ReLu(Z)
        A = dropOutRegularization(A,keepProbs[l])
        cache["A"+str(l)] = A
        
    Z_L = np.dot(parameters["W"+str(L)] , A) + parameters["b"+str(L)]
    cache["Z"+str(L)] = Z_L
    A_L = sigmoid(Z_L)
    cache["A"+str(L)] = Z_L
    
    return A_L , cache

def predictionForwardPropagation(parameters,X):  #without dropout
    
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

def computeCost(A_L , y_train, parameters ):
    m=np.shape(y_train)[0]

    if m==0 :
        print("erruer")    
    J = (1/m) * np.sum(  -np.multiply(y_train.T, np.log(A_L)) - np.multiply((1-y_train).T , np.log(1-A_L))  ) 

    return J

def derivativeCostFunction(A_L,y):
    dAL=- (np.divide(y.T, A_L) - np.divide(1 - y.T, 1 - A_L)) 
    return dAL

def backPropagation(parameters, y_train, A_L, cache):
    L = len(parameters)//2
    m = np.shape(y_train)[0]
    grads=dict()

    dAL = derivativeCostFunction(A_L ,y_train)
    dZL = dAL*derivativeSigmoid(cache["Z"+str(L)])
  
    grads["dW"+str(L)] = (1/m)* np.dot( dZL , cache["A"+str(L-1)].T )
    grads["db"+str(L)] = (1/m)* np.sum( dZL, axis=1, keepdims=True)
    dAl_1 = np.dot( parameters["W"+str(L)].T , dZL)
    
    for l in reversed(range(1,L)):
        
        dZl = dAl_1 * derivativeRelu(cache["Z"+str(l)])
        grads["dW"+str(l)] =(1/m)* np.dot( dZl,cache["A"+str(l-1)].T )
        grads["db"+str(l)] =(1/m)* np.sum( dZl, axis=1, keepdims=True)
        dAl_1 = np.dot( parameters["W"+str(l)].T , dZl )

    return grads

def updateParameters(parameters,grads,V_averageGrad,S_averageGrad, learning_rate, beta1,beta2,t,epsilon):  #adam optimization
    L = len(parameters)//2
    v_corrected ={}  #bias correction
    s_corrected ={}    
    for l in range(L):
        V_averageGrad["dW"+str(l+1)] = beta1*V_averageGrad["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]      
        V_averageGrad["db"+str(l+1)] =  beta1*V_averageGrad["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]   
        v_corrected["dW"+str(l+1)] = V_averageGrad["dW"+str(l+1)] #/ (1-np.power(beta1,t))        
        v_corrected["db"+str(l+1)] = V_averageGrad["db"+str(l+1)] #/ (1-np.power(beta1,t))

        S_averageGrad["dW"+str(l+1)] = beta2*S_averageGrad["dW"+str(l+1)] + (1-beta2)*np.power(grads["dW"+str(l+1)],2)  #/ (1 - beta2**i)        
        S_averageGrad["db"+str(l+1)] = beta2*S_averageGrad["db"+str(l+1)] + (1-beta2)*np.power(grads["db"+str(l+1)],2)  #/ (1 - beta2**i)        
        s_corrected["dW"+str(l+1)] = S_averageGrad["dW"+str(l+1)] #/ (1 - np.power(beta2,t))     
        s_corrected["db"+str(l+1)] = S_averageGrad["db"+str(l+1)] #/ (1 - np.power(beta2,t))     


        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v_corrected["dW"+str(l+1)] / np.sqrt(s_corrected["dW"+str(l+1)]+epsilon)
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v_corrected["db"+str(l+1)] / np.sqrt(s_corrected["db"+str(l+1)]+epsilon)
        


    return parameters,v_corrected,s_corrected


def plotCost(j_history,num_iters,learning_rate):
    x=np.arange(0,num_iters)
    y=j_history
    plt.plot(x,y,linewidth=2.0)
    plt.title("learning_rate =%f" %(learning_rate))
    plt.show()

def predict(X,y,parameters):
    
    A_L,cache = predictionForwardPropagation(parameters,X) #A4 : 1,200 et y: 200,1
    A_L=A_L.T
    max_indices_pred= A_L.argmax(axis=1)
    max_indices_test = y.argmax(axis=1)
  
    accuracy=np.mean(max_indices_pred == max_indices_test)
    return accuracy
        
def initAdamOptimization(parameters):
    
    V_averageGrad = dict()
    S_averageGrad = dict()
   
    for i in range(1,len(parameters)//2 +1):
        V_averageGrad["dW"+str(i)] = np.zeros(np.shape(parameters["W"+str(i)]))
        V_averageGrad["db"+str(i)] = np.zeros(np.shape(parameters["b"+str(i)]))
        S_averageGrad["dW"+str(i)] = np.zeros(np.shape(parameters["W"+str(i)]))
        S_averageGrad["db"+str(i)] = np.zeros(np.shape(parameters["b"+str(i)]))
    
    return V_averageGrad,S_averageGrad

def dropOutRegularization(activation,keepProbs):
        dropProba =  np.random.rand(np.shape(activation)[0],np.shape(activation)[1]) < keepProbs
        activation= np.multiply(dropProba,activation)  #drop few neurons
        activation = activation/keepProbs  #inverted dropOut, conservers scale
        return activation
        
        
        
# ------------   Inintialisation   -------------------------------
        
        
        
X_train, y_train, X_test, y_test = initDatasets(m_train,m_test)
X_train , X_test = standardizeDataset(X_train,X_test)
 
#plotImage(X_train,0,32)

parameters = initialiseParameters(networkArch)

V_averageGrad,S_averageGrad = initAdamOptimization(parameters)

j_history=np.zeros((num_iters,1))
J=0
iteration=0
t=0

print("Deep neural network\nMini-batch grad descent with momentum and drop out regularization")

pbar = tqdm(range(1,num_iters))

for iter in pbar:
    for batch_num in range(1, int( m_train//mini_batch_size + int(m_train%mini_batch_size >= 1))+1 ):
        
        if (batch_num)*mini_batch_size <= m_train:  #si depasse pas X_train

            X_mini_batch = X_train[ :, (batch_num-1)*mini_batch_size : (batch_num)*mini_batch_size]
            y_mini_batch = y_train[(batch_num-1)*mini_batch_size : (batch_num)*mini_batch_size , :] 
        else: #si oui, dernier batch avec ce qui reste

            X_mini_batch = X_train[:,(batch_num-1)*mini_batch_size:]
            y_mini_batch = y_train[(batch_num-1)*mini_batch_size: , :]
            
        A_L , cache = dropOutforwardPropagation(parameters,X_mini_batch,keepProbs)
        J = computeCost(A_L,y_mini_batch,parameters)
        
        grads = backPropagation(parameters,y_mini_batch,A_L,cache) 
        t=t+1
        parameters,V_averageGrad,S_averageGrad=updateParameters(parameters,grads,V_averageGrad,S_averageGrad, learning_rate, beta1,beta2,batch_num,epsilon)
        
    pbar.set_description("cost iter %f"  %J)
    
    j_history[iter,0] = J

print(np.shape(j_history))
    
accuracy_train=predict(X_train,y_train,parameters)
accuracy_test=predict(X_test,y_test,parameters)

print("alpha=%f, trainSet:%f testSet:%f" %(learning_rate,accuracy_train, accuracy_test))
plotCost(j_history,num_iters,learning_rate)
