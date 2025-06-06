'''
Deep network tools
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
from keras.layers import InputLayer, Dense, Dropout
from keras.models import Sequential

def deep_network_basic(n_inputs, n_hidden, n_output, activation='elu', activation_out=None,
                      lrate=0.001, opt=None, loss='mse', dropout=None, dropout_input=None, 
                       kernel_regularizer=None, kernel_regularizer_L1=None,
                       metrics=None):

    if dropout is not None:
        print("DENSE: DROPOUT %f"%dropout)
    if kernel_regularizer is not None:
        print("DENSE: L2 Regularization %f"%kernel_regularizer)
        kernel_regularizer=keras.regularizers.l2(kernel_regularizer)
    elif kernel_regularizer_L1 is not None:
        # Only us L1 if specified *and* L2 is not active
        print("DENSE: L1 Regularization %f"%kernel_regularizer_L1)
        kernel_regularizer=keras.regularizers.l1(kernel_regularizer_L1)
            
    # TODO: add other loss functions
    model = Sequential()
    
    # Input layer
    model.add(InputLayer(shape=(n_inputs,)))
    
    # Dropout input features?
    if dropout_input is not None:
            model.add(Dropout(rate=dropout_input, name="dropout_input"))
            
    # Loop over hidden layers
    #i = 0
    for i, n in enumerate(n_hidden):             
        model.add(Dense(n, use_bias=True, name="hidden_%02d"%(i), activation=activation,
                 kernel_regularizer=kernel_regularizer))
        
        if dropout is not None:
            model.add(Dropout(rate=dropout, name="dropout_%02d"%(i)))
            
        #i += 1
    
    # Output layer
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation_out))
    
    # Default optimizer
    if opt is None:
        opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
        #beta_1=0.9, beta_2=0.999,
        #epsilon=None, decay=0.0, amsgrad=False)
        
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model

def create_dense_stack(tensor, n_hidden, lambda_regularization=None,
                       name='D', activation='elu', dropout=0.5,
                       name_last='output',
                       activation_last='elu',
                       batch_normalization=False):

    if isinstance(lambda_regularization, (int, float)):
        lambda_regularization=keras.regularizers.l2(lambda_regularization)
    
    for n, i in zip(n_hidden, range(len(n_hidden)-1)):
        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        tensor = Dense(n, use_bias=True,
                       bias_initializer='zeros',
                       name="%s_%d"%(name,i),
                       activation=activation,
                       kernel_regularizer=lambda_regularization,
                       kernel_initializer='truncated_normal')(tensor)
        if dropout is not None:
            tensor = Dropout(rate=dropout,
                             name="%s_drop_%d"%(name,i))(tensor)
                       

    # Last layer
    if batch_normalization:
        tensor = BatchNormalization()(tensor)

    tensor = Dense(n_hidden[-1], use_bias=True,
                   bias_initializer='zeros',
                   name=name_last,
                   activation=activation_last,
                   kernel_regularizer=lambda_regularization,
                   kernel_initializer='truncated_normal')(tensor)
    
    
    return tensor
