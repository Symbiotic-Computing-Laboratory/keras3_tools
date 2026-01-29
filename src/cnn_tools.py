'''
Andrew H. Fagg

CNN-based classifiers

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random

import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, BatchNormalization
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Concatenate
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, SpatialDropout2D
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, SpatialDropout3D
from fully_connected_tools import *

class ConvolutionalNeuralNetwork:
    conv_operators = {'C1': (Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D),
                      'C2': (Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, SpatialDropout2D),
                      'C3': (Convolution3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, SpatialDropout3D),
                      }

    @staticmethod
    def create_cnn_stack(tensor,
                         conv_filters:[int],
                         conv_kernel_size:[int],
                         conv_pool_average:[int],
                         conv_pool:[int],
                         conv_strides:[int],
                         conv_type:str='C2',
                         name_base:str='',
                         conv_activation:str='elu',
                         regularizer=None,
                         s_dropout=None,
                         padding='valid',
                         batch_normalization=False,
                         global_max_pool=False):
        '''
        Create a linear convolutional stack
        
        '''
        # Access the layer constructors for the dimensionality of the input
        conv, mp, ap, gmp, sd = ConvolutionalNeuralNetwork.conv_operators[conv_type]

        # Handle custom activation function strings
        conv_activation = FullyConnectedNetwork.translate_activation_function(conv_activation)


        # Deal with case where conv_strides is None (translate to all 1s)
        if conv_strides is None:
            conv_strides = [1]*min([len(conv_filters), len(conv_kernel_size), len(conv_pool)])

        # Also deal with conv_pool_average being None
        if conv_pool_average is None:
            conv_pool_average = [None]*min([len(conv_filters), len(conv_kernel_size), len(conv_pool)])

        # Loop over all convolutional layers
        for i, (f,k,avg_p,p,s) in enumerate(zip(conv_filters, conv_kernel_size, conv_pool_average, conv_pool, conv_strides)):
            
            # Average pooling
            if avg_p is not None and avg_p > 1:
                tensor = ap(avg_p,
                            strides=avg_p,
                            padding=padding)(tensor)

            # Convolution
            tensor = conv(filters=f,
                          kernel_size=k,
                          strides=s,
                          padding=padding,
                          use_bias=True,
                          kernel_initializer='truncated_normal',
                          bias_initializer='zeros',
                          name='%s%d'%(name_base,i),
                          activation=conv_activation,
                          kernel_regularizer=regularizer)(tensor)
        
            # Possible dropout
            if s_dropout is not None:
                tensor = sd(s_dropout)(tensor)

            # Batch Norm
            if batch_normalization:
                tensor = BatchNormalization()(tensor)

            # Max pooling
            if p is not None and p > 1:
                tensor = mp(p,
                            strides=p,
                            padding=padding)(tensor)

        # Tack on global max pooling
        if global_max_pool:
            tensor=gmp()(tensor)
            
        return tensor


    @staticmethod
    def create_cnn_network(input_shape=None,
                           input_dtype=None,
                           conv_kernel_size=[3,3],
                           conv_padding='valid',
                           conv_number_filters=[8,16],
                           conv_activation='elu',
                           conv_pool_average_size=None,
                           conv_pool_size=[0,2],
                           conv_strides=None,
                           spatial_dropout=None,
                           conv_batch_normalization=False,
                           n_hidden=[10,2],
                           output_shape=[1],
                           dropout_input=False,
                           name_base='',
                           activation='elu',
                           lambda1:float=None,
                           lambda2:float=None,
                           dropout=None,
                           name_last='output',
                           activation_last=None,
                           batch_normalization=False,
                           learning_rate=0.001,
                           loss='mse',
                           metrics=[],
                           opt=None,
                           **kwargs):

        ##################
        # Less frequentyly used arguments are in kwargs

        # Support for tokenizer and embedding layers
        tokenizer = kwargs.pop("tokenizer", False)
        embedding = kwargs.pop("embedding", False)
        tokenizer_max_tokens = kwargs.pop("tokenizer_max_tokens", None)
        tokenizer_standardize = kwargs.pop("tokenizer_standardize", "lower_and_strip_punctuation")
        tokenizer_split = kwargs.pop("tokenizer_split", "whitespace")
        tokenizer_output_sequence_length = kwargs.pop("tokenizer_output_sequence_length", None)
        tokenizer_vocabulary = kwargs.pop("tokenizer_vocabulary", None)
        tokenizer_encoding = kwargs.pop("tokenizer_encoding", "utf-8")
        embedding_dimensions = kwargs.pop("embedding_dimensions", None)
        ##################

        model_text_vectorization = None

        
        # Infer input dtype
        if input_dtype is None:
            input_dtype = "float32"

            if tokenizer:
                input_dtype = "string"
                
            elif embedding:
                input_dtype = "int32"
        
        ##################

        # TODO: check input_shape form
        input_len = len(input_shape)
        assert (input_len >= 2 and input_len <= 4) or (input_len == 1 and tokenizer), "CNN Input Shape must have dimensionality 2, 3, or 4"
    
        # TODO: check output_shape form

        activation = FullyConnectedNetwork.translate_activation_function(activation)
        activation_last = FullyConnectedNetwork.translate_activation_function(activation_last)

        # Convolution dimensionality
        # Tokenizer only supports 1D.  Otherwise, infer from input shape

        if tokenizer:
            conv_type = 'C1'
        else:
            conv_type = ['C1', 'C2', 'C3'][len(input_shape)-2]
            
        print("Convolution type:", conv_type)
    
        # Resolve regularizer
        regularizer = FullyConnectedNetwork.create_regularizer(lambda1, lambda2)

        # Input layer
        input_tensor = tensor = Input(shape=input_shape,
                                      name=name_base + 'input',
                                      dtype=input_dtype)

        # Optional tokenizer
        if tokenizer:
            # Translation from strings to ints
            model_text_vectorization = keras.layers.TextVectorization(max_tokens=tokenizer_max_tokens,
                                                    standardize=tokenizer_standardize,
                                                    split=tokenizer_split,
                                                    output_sequence_length=tokenizer_output_sequence_length,
                                                    vocabulary=tokenizer_vocabulary,
                                                    encoding=tokenizer_encoding,
                                                    )
            tensor = model_text_vectorization(tensor)

        # Embedding required if we have a tokenizer, but can have embedding without tokenizer
        if tokenizer or embedding:
            # Translation from int tokens to embeddings
            tensor = keras.layers.Embedding(
                input_dim=tokenizer_max_tokens,
                output_dim=embedding_dimensions,
                )(tensor)


        # Dropout input features?
        #if dropout_input is not None:
        #tensor = Dropout(rate=dropout_input, name=name_base+"dropout_input")(tensor)
        tensor = ConvolutionalNeuralNetwork.create_cnn_stack(tensor,
                                                             conv_filters=conv_number_filters,
                                                             conv_kernel_size=conv_kernel_size,
                                                             conv_pool_average=conv_pool_average_size,
                                                             conv_pool=conv_pool_size,
                                                             conv_strides=conv_strides,
                                                             conv_type=conv_type,
                                                             name_base=name_base + '_C',
                                                             conv_activation=conv_activation,
                                                             regularizer=regularizer,
                                                             s_dropout=spatial_dropout,
                                                             padding=conv_padding,
                                                             batch_normalization=conv_batch_normalization,
                                                             global_max_pool=True)

        # Create the rest of the network
        tensor = FullyConnectedNetwork.create_dense_stack(tensor=tensor,
                                                          n_hidden=n_hidden,
                                                          activation=activation,
                                                          regularizer=regularizer,
                                                          dropout=dropout,
                                                          name=name_base+'_FC',
                                                          #name_last=name_last,
                                                          #activation_last=activation_last,
                                                          batch_normalization=batch_normalization)

        # Output layer
        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        if len(output_shape) == 1:
            # Output is 1D
            tensor = Dense(output_shape[0],
                           use_bias=True,
                           bias_initializer='zeros',
                           name=name_base + 'output',
                           activation=activation_last,
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)
        else:
            # Output is kD, where k > 1
            # Linear transformation from previous layer to one that is the product of
            #   all the output dimensions, then reshape, then non-linearity
            #
            #  This feature is useful for cases such as outputing K probability distributions for
            #   which we are using softmax as the non-linearity.
            
            tensor = Dense(reduce(operator.mul, output_shape, 1),
                           use_bias=True,
                           bias_initializer='zeros',
                           name=name_base + 'output_vector',
                           activation='linear',
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)
            tensor = Reshape(output_shape)(tensor)
            tensor = Activation(activation_last)(tensor)


        if opt is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                        amsgrad=False)

        # Create the model
        model = Model(input_tensor, tensor)
    
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        if model_text_vectorization is None:
            return model
        else:
            return model, model_text_vectorization
    
