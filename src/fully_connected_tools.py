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

# Keras3
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model

class FullyConnectedNetwork:
    
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
                           n_hidden,
                           name='D',
                           activation='elu',
                           regularizer = None,
                           dropout=None,
                           #name_last='output',
                           #activation_last=None,
                           batch_normalization=False):

        for i, n in enumerate(n_hidden):
            if batch_normalization:
                tensor = BatchNormalization()(tensor)

            tensor = Dense(n, use_bias=True,
                           bias_initializer='zeros',
                           name="%s_%d"%(name,i),
                           activation=activation,
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)
            if dropout is not None:
                tensor = Dropout(rate=dropout,
                                 name="%s_dropout_%d"%(name,i))(tensor)
                       
        return tensor


    @staticmethod
    def create_fully_connected_network(input_shape=None,
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
                                       opt=None):

        # TODO: check input_shape form
        # TODO: check output_shape form

        regularizer = FullyConnectedNetwork.create_regularizer(lambda1, lambda2)

        # Input layer
        input_tensor = tensor = Input(shape=(input_shape[0],),
                                      name=name_base + 'input')
    
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

        tensor = Dense(output_shape[0],
                       use_bias=True,
                       bias_initializer='zeros',
                       name=name_base + 'output',
                       activation=activation_last,
                       kernel_regularizer=regularizer,
                       kernel_initializer='truncated_normal')(tensor)

        if opt is None:
            # Default optimizer
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                        amsgrad=False)

        # Create the model
        model = Model(input_tensor, tensor)
    
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        return model
    

            

