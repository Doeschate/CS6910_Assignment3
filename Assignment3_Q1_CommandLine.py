#importing required Libraries
import os, shutil, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten,Embedding,Dense,Concatenate
from keras.utils.vis_utils import plot_model


#Choose the language
# Colab
# dir = '/content/dakshina_dataset_v1.0'
# Kaggle
dir = './dakshina_dataset_v1.0'
lang = "hi"

#Removing the unnecessary folders and keeping only the lang choosen lexicons folder
for files in os.listdir(dir):
    if (files==lang):    
        path = os.path.join(dir, files)
        for files_lang in os.listdir(path):
            if(files_lang!="lexicons"):
                path_lang = os.path.join(path, files_lang)
                try:
                    shutil.rmtree(path_lang)
                except OSError:
                    os.remove(path_lang)
    else:    
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

#Train, validation and test dataset path
train_folder = ".translit.sampled.train.tsv"
train_path = os.path.join(dir, lang,"lexicons",lang+train_folder)

# Parsing the training data to find number of unique characters in input language and output language( these values are useful in creating embedding layer)
# readData function will take path as argument and return the data present in that path as pandas DataFrame
def readData(path):    
    trainingData_df = pd.read_csv(path, sep='\t',on_bad_lines='skip',header=None)
    trainingData = trainingData_df.values.tolist()
    return trainingData

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Reading the training Data as data frame
trainingData = readData(train_path)

# Iterating through each training data item
for line in trainingData:
    input_text, target_text = line[1],line[0]
    if not isinstance(input_text,str):
        continue
    target_text = " " + target_text + " "
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
# input_characters list contain unique characters in input language 
input_characters.add(' ')
# target_characters list contain unique characters in target language 
target_characters.add(' ')
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))


num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# max_encoder_seq_length is maximum length word in input language
max_encoder_seq_length = max([len(txt) for txt in input_texts])
# max_decoder_seq_length is maximum length word in target language
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

'''
builModel will take 
    
    latent_dims - list of hidden states sizes, list length gives number of encoders and decoder present in the model
    EmbeddingOutputDimensions - [encoder embedding output size, decoder embedding output size]
    layer_type - type of cell used in encoder and decoder (rnn or lstm or gru)
    
    as arguments and return prepared model
'''

def buildModel(latent_dims,EmbeddingOutputDimensions,layer_type):
    encoder_inputs = keras.Input(shape=(None,)) # input layer
    embedding_encoder_layer = Embedding(input_dim = num_encoder_tokens, output_dim = EmbeddingOutputDimensions[0], trainable=True) # embedding layer
    embedding_encoder_inuts = embedding_encoder_layer(encoder_inputs)
    encoder_outputs = embedding_encoder_inuts
    
    encoder_states = [] # encoder_states stores last hidden states return by each cell in encoder to provide as input to respective decoder cells 
    
    # this loop creats layer_type cells with configuration present in latent_dims list
    for i in range(len(latent_dims))[::-1]:
        if layer_type == 'lstm':
            encoder_outputs, state_h, state_c = keras.layers.LSTM(latent_dims[i], return_state=True,return_sequences=True)(encoder_outputs)
            encoder_states += [state_h, state_c]
        if layer_type == 'gru':
            encoder_outputs, state_h= keras.layers.GRU(latent_dims[i], return_state=True, return_sequences=True)(encoder_outputs)
            encoder_states += [state_h]
        if layer_type == 'rnn':
            encoder_outputs, state_h = keras.layers.SimpleRNN(latent_dims[i], return_state=True, return_sequences=True)(encoder_outputs)
            encoder_states += [state_h]

    decoder_inputs = keras.Input(shape=(None,)) # input layer
    embedding_decoder_layer = Embedding(input_dim = num_decoder_tokens, output_dim = EmbeddingOutputDimensions[1],trainable=True) # embedding layer
    embedding_decoder_inputs = embedding_decoder_layer(decoder_inputs)
    decoder_outputs_temp = embedding_decoder_inputs
    
    # this loop creats layer_type cells with configuration present in latent_dims list
    for i in range(len(latent_dims)):
        if layer_type == 'lstm':
            decoder_outputs_temp, dh, dc = keras.layers.LSTM(latent_dims[len(latent_dims) - i - 1], return_sequences=True, return_state=True)(decoder_outputs_temp, initial_state=encoder_states[2*i:2*(i+1)])
        if layer_type == 'gru':
            decoder_outputs_temp,dh = keras.layers.GRU(latent_dims[len(latent_dims) - i - 1], return_sequences=True, return_state=True)(decoder_outputs_temp, initial_state=encoder_states[i])
        if layer_type == 'rnn':
            decoder_outputs_temp, dh = keras.layers.SimpleRNN(latent_dims[len(latent_dims) - i - 1], return_sequences=True, return_state=True)(decoder_outputs_temp, initial_state=encoder_states[i])
     
    dense_layer = keras.layers.Dense(num_decoder_tokens, activation="softmax") #dense layer
    decoder_outputs = dense_layer(decoder_outputs_temp) 
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs) # creating model
    
    return model


if  __name__ =="__main__":
    '''
    Format of command line arguments
    1) number of cells in encoder or decoder (let n)
    2 to n) hidden states sizes
    n+1) encoder embedding output size
    n+2) decoder embedding output size
    n+3) cell type (rnn or lstm or gru)
    '''

    number_of_en_de = int(sys.argv[1])
    latent_dims = []
    for i in range(0,number_of_en_de):
        latent_dims.append(int(sys.argv[i+2]))
    embed_dims = [int(sys.argv[number_of_en_de+2]),int(sys.argv[number_of_en_de+3])]
    cell_type = sys.argv(number_of_en_de+4)


    # build model
    model = buildModel(latent_dims=latent_dims,EmbeddingOutputDimensions=embed_dims,layer_type=cell_type)
    model.summary()