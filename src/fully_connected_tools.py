'''
Fully connected network tools

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time

import argparse
import pickle
from functools import reduce
import operator

# Keras3
from keras.layers import Input, Dense, Dropout, Reshape, Activation, BatchNormalization
from keras.models import Sequential, Model
import keras.saving as saving
import keras.ops as ops

@saving.register_keras_serializable(package="zero2neuro")
def elup1(x):
    # elu(x) + 1
    return ops.elu(x)+1.0

class FullyConnectedNetwork:

    @staticmethod
    def translate_activation_function(name:str):
        if isinstance(name, str):
            if name == 'elup1':
                return elup1
            else:
                return name
        else:
            return name
        
    @staticmethod
    def create_regularizer(lambda1:float=None, lambda2:float=None):
        if lambda1 is None:
            if lambda2 is None:
                return None
            else:
                return keras.regularizers.l2(lambda2)
        else:
            if lambda2 is None:
                return keras.regularizers.l1(lambda1)
            else:
                return keras.regularizers.l1l2(lambda1, lambda2)

    @staticmethod
    def create_dense_stack(tensor,
                           n_hidden=None,
                           name='D',
                           activation='elu',
                           regularizer = None,
                           dropout=None,
                           batch_normalization=False):

        # Handle custom activation strings
        activation = FullyConnectedNetwork.translate_activation_function(activation)

        # Iterate over each dense module
        for i, n in enumerate(n_hidden):
            # Fully connected layer
            tensor = Dense(n, use_bias=True,
                           bias_initializer='zeros',
                           name="%s_%d"%(name,i),
                           activation=activation,
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)

            # Dropout
            if dropout is not None:
                tensor = Dropout(rate=dropout,
                                 name="%s_dropout_%d"%(name,i))(tensor)

            # Batch normalization
            if batch_normalization:
                tensor = BatchNormalization()(tensor)

        return tensor


    @staticmethod
    def create_fully_connected_network(input_shape=None,
                                       input_dtype=None,
                                       batch_normalization_input=False,
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
        # TODO: check output_shape form

        activation = FullyConnectedNetwork.translate_activation_function(activation)
        activation_last = FullyConnectedNetwork.translate_activation_function(activation_last)

        regularizer = FullyConnectedNetwork.create_regularizer(lambda1, lambda2)

        # Input layer
        print("INPUT SHAPE", str(input_shape))
        #input_tensor = tensor = Input(shape=input_shape,
        input_tensor = tensor = Input(shape=(input_shape[0],),
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

            # Flatten so that we can interface to the rest of the FC network
            tensor = keras.layers.Flatten()(tensor)
            

        # Batch Normalize the inputs
        if batch_normalization_input:
            tensor = BatchNormalization()(tensor)
            
        # Dropout input features?
        if dropout_input is not None:
            tensor = Dropout(rate=dropout_input, name=name_base+"dropout_input")(tensor)

        # Create the rest of the network
        tensor = FullyConnectedNetwork.create_dense_stack(tensor=tensor,
                                                          n_hidden=n_hidden,
                                                          activation=activation,
                                                          regularizer=regularizer,
                                                          dropout=dropout,
                                                          name='hidden',
                                                          # name_last=name_last,
                                                          # activation_last=activation_last,
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
            # Default optimizer
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                            amsgrad=False)

        # Create the model
        model = Model(input_tensor, tensor)
    
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        if model_text_vectorization is None:
            return model
        else:
            return model, model_text_vectorization
    

