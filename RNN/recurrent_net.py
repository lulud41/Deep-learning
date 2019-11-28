#!/usr/bin/env python3


"""
	Inteligencia Artificial

	DEROUET Lucien Emile

	Proyecto : Uso de redes recurrentes para la clasificacion binaria de comentarios de peliculas
	de la base de datos de IMDB

	Los comentarios son positivos (label =1) o negativos (label =0)

	Para correr el escript se necesita : numpy, tensorflow, keras, matplotlib, 
		los datos de entrenamiento (base IMDB en raw text) y los pesos pre entranados del GloVe word embedding


"""

import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN,Dense

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATASET_FOLDER = "aclImdb"
EMBEDDING_WEIGHTS_FILE = "glove.6B.100d.txt"
EMBEDDING_SIZE = 100  #size of encoded vectors

TRAIN_SET_SIZE = 30000
TEST_SET_SIZE = 10000

review_max_len = 100  #max size of time steps (words) allowed for a sequence
total_words_number = 10000 # consider only the 10 000 top words

learning_rate = 0.001
num_epochs = 20
batch_size=128

"""
	Create a raw text list and labels list from the original unenconded imdb dataset

"""
def dataset_initializer():

	if DATASET_FOLDER not in os.listdir():
		print("IMDB raw text dataset has to be downloaded !\n\
			\t> find it at : mng.bz/0tIo\n\
			\t> extract the data in a folder called : "+str(DATA_FOLDER))
		exit()

	#if IMDB exists : load data

	raw_text_data = []
	labels = []

	for data_folder in ["test","train"]:
		for data_label in ["neg","pos"]:
			for file in os.listdir(os.path.join(DATASET_FOLDER,data_folder,data_label)):
				if file[-4:] == ".txt":
					f = open(os.path.join(DATASET_FOLDER,data_folder,data_label,file))
					raw_text_data.append(f.read())
					if data_label == "neg":
						labels.append(0)
					if data_label == "pos":
						labels.append(1)

	print(str(len(raw_text_data))+" movie reviews found ! ")

	return raw_text_data,labels

"""
	give a numerical value to every word (token)
	Cuts every sequence longer than the max size and padds the smaller ones
	Returns a padded sequences matrix and the word indexes dictionnary {"word": index}
"""
def tokenize_raw_data(raw_data):
	tokenizer = Tokenizer(num_words=total_words_number)
	tokenizer.fit_on_texts(raw_data)
	sequences = tokenizer.texts_to_sequences(raw_data)
	word_indexes = tokenizer.word_index
	
	padded_sequences = pad_sequences(sequences,review_max_len)
	print(padded_sequences.shape)
	print("> End of tokenization")
	return padded_sequences,word_indexes

"""
	Seperate the training and test set from the tokenized data

"""
def prepare_test_train_sets(padded_sequences,labels):
	rand_perm = np.arange(padded_sequences.shape[0])
	np.random.shuffle(rand_perm)

	labels = np.asarray(labels)
	labels = labels[rand_perm]
	padded_sequences = padded_sequences[rand_perm,:]

	x_train = padded_sequences[0:TRAIN_SET_SIZE,:]
	y_train = labels[0:TRAIN_SET_SIZE]

	x_test = padded_sequences[TRAIN_SET_SIZE:TRAIN_SET_SIZE+TEST_SET_SIZE+1,:]
	y_test = labels[TRAIN_SET_SIZE : TRAIN_SET_SIZE+TEST_SET_SIZE+1]
	print("> train and test sets ready !")
	print("x_train shape : "+str(x_train.shape))
	print("tokens of first sequence : "+str(x_train[0]))
	return x_train,y_train,x_test,y_test

"""
	Realize the GloVe word embedding matrix
	Matrix of shape : (nb_of_differents_words , size_of_encoded_vector )

	The matrix maps every word index to a n dimensional dense vector, that can enter the RNN

"""
def word_embedding_matrix(word_indexes):

	if EMBEDDING_WEIGHTS_FILE not in os.listdir():
		print("> Error : GloVe pretrained weights needed, please download the file at \
	\n: https://nlp.stanford.edu/projects/glove/")
		exit()
	f = open(EMBEDDING_WEIGHTS_FILE)

	embedding_matrix = np.zeros((total_words_number, EMBEDDING_SIZE))

	for line in f:
		vect_of_file = line.split()
		if vect_of_file[0] in word_indexes.keys() and word_indexes.get(vect_of_file[0]) < total_words_number:
			embedded_vect = np.asarray(vect_of_file[1:],dtype="float32")
			embedding_matrix[word_indexes.get(vect_of_file[0]),:] = embedded_vect

	print("exemple of encoding :\
		\nfor the word 'home', index ="+str(word_indexes.get("home"))+"\
		\nencoding : "+str(embedding_matrix[word_indexes.get("home")]))

	print("shape embedding_matrix "+str(embedding_matrix.shape))

	return embedding_matrix

"""
	Init the Keras RNN model, with two possible architectures :

	> arch1 : one hidden reccurent layer
	> arch2 : two hidden reccurent layers
"""
def init_model(embedding_matrix, architecture=2):

	model = Sequential()
	model.add(Embedding(total_words_number,EMBEDDING_SIZE,input_length=review_max_len))
	
	model.layers[0].trainable=False
	model.layers[0].set_weights([embedding_matrix])
	
	 #arch 1 : one hidden reccurent layer
	if architecture == 1:         
		model.add(SimpleRNN(32))

	# arch 2 : two hidden reccurents layers
	elif architecture == 2:  
		model.add(SimpleRNN(32,return_sequences=True))
		model.add(SimpleRNN(32))

	model.add(Dense(1,activation = 'sigmoid'))
	model.summary()

	return model

"""
	Train the model with the training set, with the RMSprop optimizer anc accuracy metric
	It uses the EarlyStopping callback : stops training when the performance stops increasing
	and the ModelCheckpoint to save the model at the end of training
	
	Returns the history object, contaiing the training data : accuracy and loss history
"""
def train_model(model,x_train,y_train):

	model.compile(loss='binary_crossentropy',
		optimizer=RMSprop(lr=learning_rate),
		metrics=["acc"])
	
	call_backs_list = [
    	EarlyStopping(
	        monitor='acc', patience = 0),

	    ModelCheckpoint(
	        filepath = 'rnn_model_checkpoint.h5',
	        monitor = 'val_loss',
	        save_best_only = True,)	
	    ]

	history = model.fit(x_train,y_train,epochs=num_epochs,
		batch_size=batch_size,validation_split=0.2,
		callbacks = call_backs_list)

	return history
"""
	Test the model performance on the test data
"""
def test_model(x_test,y_test,model):

	perf = model.evaluate(x_test, y_test, batch_size=batch_size)
	print('Model evaluation : test loss, test acc:', perf)


"""
	Plot the loss and accuracy curves from the history data
"""
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
    
    print("accuracy on trainSet : ",acc[len(acc)-1]," accuracy on val_Set : ", val_acc[len(val_acc)-1] )
    
    plt.show()
 
if __name__ == '__main__':

	raw_data, labels = dataset_initializer()
	padded_sequences,word_indexes = tokenize_raw_data(raw_data)
	
	x_train,y_train,x_test,y_test = prepare_test_train_sets(padded_sequences,labels)
	embedding_matrix = word_embedding_matrix(word_indexes)
	model = init_model(embedding_matrix,architecture=2)
	
	history = train_model(model,x_train,y_train)
	
	plot_cost(history)
	test_model(x_test,y_test,model)
